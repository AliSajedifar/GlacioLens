"""Build and inspect exactly the reviewed allowlist. Uses only the standard library."""
import argparse
import ast
import hashlib
import json
import os
import re
import stat
import struct
import zipfile
import zlib
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "glacial_lake_detection/"
ALLOWED_SUFFIXES = {".py", ".md", ".json", ".csv", ".toml", ".txt", ".png"}
# Reviewed metadata only; keep the default small-member limit for every other path.
MEMBER_SIZE_LIMITS = {
    "data/manifests/source_members.csv": 7_000_000,
    "data/manifests/auxiliary_pool.json": 1_500_000,
    "data/manifests/auxiliary_plan.json": 2_500_000,
}
ALLOWED_DOTFILES = {".gitignore", ".gitattributes", "LICENSE"}
BLOCKED_PARTS = {"_local", ".git", ".cache", ".tools", ".venv-mcp", ".codex", "__pycache__", "user_assets"}
PATTERNS = {
    "private-key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "credential-token": r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|AIza[A-Za-z0-9_-]{30,}|AKIA[A-Z0-9]{16}|ya29\.[A-Za-z0-9_-]{20,})\b",
    "workstation-path": r"(?i)(?:[A-Z]:[\\/](?:Users|Research)[\\/]|/content/" + r"drive/|/home/[^ /]+/)",
    "private-storage-url": r"https?://(?:drive\.google\.com|colab\.research\.google\.com|docs\.google\.com)/",
    "private-routing-field": r'"(?:file_id|folder_id|notebook_id|session_id|refresh_token|client_secret)"\s*:\s*"[^"\s]+"',
    "assigned-secret": r'(?i)(?:api_key|access_token|password|secret_key)\s*[:=]\s*["\x27][A-Za-z0-9_+./=-]{16,}["\x27]',
}


def safe_path(relative):
    rel = PurePosixPath(relative)
    if not relative or rel.is_absolute() or ".." in rel.parts or ":" in relative or "\\" in relative:
        raise ValueError("Unsafe relative path")
    if set(rel.parts) & BLOCKED_PARTS:
        raise ValueError("Private or generated path in allowlist")
    p = ROOT
    for part in rel.parts:
        p = p / part
        s = p.lstat()
        if stat.S_ISLNK(s.st_mode) or getattr(s, "st_file_attributes", 0) & (0x400 | 0x1000 | 0x40000 | 0x400000):
            raise ValueError("Linked, offline or reparse member rejected")
    if not p.is_file() or p.resolve().parent != p.parent.resolve():
        raise ValueError("Nonregular export member")
    if p.suffix not in ALLOWED_SUFFIXES and relative not in ALLOWED_DOTFILES:
        raise ValueError("Unexpected file type")
    if p.stat().st_size > MEMBER_SIZE_LIMITS.get(relative, 1_000_000):
        raise ValueError("Unexpected large member")
    return p


def findings(name, data, terms):
    if name.endswith(".png"):
        # Only a small raster preview with no textual/EXIF/private ancillary chunks.
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Invalid PNG signature")
        offset, chunks = 8, []
        while offset < len(data):
            if offset + 12 > len(data):
                raise ValueError("Truncated PNG")
            length = struct.unpack(">I", data[offset:offset+4])[0]
            kind = data[offset+4:offset+8]
            end = offset + 12 + length
            if end > len(data) or kind not in {b"IHDR", b"IDAT", b"IEND"}:
                raise ValueError("Unexpected PNG content or metadata")
            body = data[offset+8:end-4]
            if zlib.crc32(kind + body) != struct.unpack(">I", data[end-4:end])[0]:
                raise ValueError("PNG chunk CRC mismatch")
            chunks.append(kind)
            if kind == b"IHDR":
                if length != 13:
                    raise ValueError("Invalid PNG header")
                w, h, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", body)
                if not (0 < w <= 2000 and 0 < h <= 2000 and depth == 8 and color in (2, 6)
                        and compression == filtering == interlace == 0):
                    raise ValueError("Unexpected PNG geometry or encoding")
            if kind == b"IEND" and length:
                raise ValueError("Invalid PNG end")
            offset = end
        if not (chunks and chunks[0] == b"IHDR" and chunks[-1] == b"IEND"
                and chunks.count(b"IHDR") == chunks.count(b"IEND") == 1
                and all(k == b"IDAT" for k in chunks[1:-1]) and len(chunks) >= 3):
            raise ValueError("Invalid PNG chunk order")
        return []
    text = data.decode("utf-8")
    if "\x00" in text:
        raise ValueError("Binary content rejected")
    found = [{"path": name, "type": key, "value": "[REDACTED]"}
             for key, regex in PATTERNS.items() if re.search(regex, text)]
    for term in terms:
        if term in text:
            found.append({"path": name, "type": "known-private-identifier", "value": "[REDACTED]"})
            break
    if name.endswith(".json"):
        json.loads(text)
    if name.endswith(".py"):
        ast.parse(text, filename=name)
    return found


def publishing_tree(dest, payload):
    """Materialize only reviewed bytes; refuse unexpected content instead of deleting it."""
    tree = dest / "github_source"
    expected = {n for n, _ in payload}
    expected_dirs = {str(p) for n in expected for p in PurePosixPath(n).parents if str(p) != "."}
    if tree.exists():
        pending = [tree]
        while pending:
            directory = pending.pop()
            if directory.is_symlink() or getattr(directory.lstat(), "st_file_attributes", 0) & 0x400:
                raise ValueError("Linked publishing tree rejected")
            with os.scandir(directory) as entries:
                for entry in entries:
                    p = Path(entry.path)
                    s = p.lstat()
                    rel = p.relative_to(tree).as_posix()
                    if stat.S_ISLNK(s.st_mode) or getattr(s, "st_file_attributes", 0) & (0x400 | 0x1000 | 0x40000 | 0x400000):
                        raise ValueError("Linked or offline publishing member rejected")
                    if stat.S_ISDIR(s.st_mode) and rel in expected_dirs:
                        pending.append(p)
                    elif not (stat.S_ISREG(s.st_mode) and rel in expected):
                        raise ValueError("Unexpected publishing tree content; inspect before changing it")
    tree.mkdir(exist_ok=True)
    for name, data in payload:
        p = tree.joinpath(*PurePosixPath(name).parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    members = [{"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
               for name, data in payload]
    actual = sorted(p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file())
    assert actual == sorted(expected)
    assert all((tree / n).read_bytes() == data for n, data in payload)
    identity = hashlib.sha256(json.dumps(members, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"path": "dist/github_source", "files": len(members), "members_sha256": identity,
            "identity_algorithm": "SHA256 of UTF-8 canonical JSON (sorted keys, compact separators) of ordered relative path/bytes/sha256 records",
            "exact_archive_payload_match": True, "no_git_history": True}


def build(private_terms=None):
    allow = json.loads((ROOT / "release/public_files.json").read_text(encoding="utf-8"))["files"]
    if allow != sorted(set(allow)) or "release/public_files.json" not in allow:
        raise ValueError("Allowlist must be sorted, unique and include itself")
    terms = []
    if private_terms:
        known = json.loads(Path(private_terms).read_text(encoding="utf-8"))
        terms = sorted(set(known["identifiers"] + known["artifact_paths"]))
    payload = [(name, safe_path(name).read_bytes()) for name in allow]
    errors = [r for n, data in payload for r in findings(n, data, terms)]
    if errors:
        raise ValueError(json.dumps({"redacted_findings": errors}))
    dest = ROOT / "dist"
    if dest.exists() and (dest.is_symlink() or getattr(dest.lstat(), "st_file_attributes", 0) & 0x400):
        raise ValueError("Linked output directory rejected")
    dest.mkdir(exist_ok=True)
    output = dest / "glacial_lake_detection_source.zip"
    manifest_path = dest / "glacial_lake_detection_source.manifest.json"
    for target in (output, manifest_path):
        if target.exists() and (target.is_symlink() or getattr(target.lstat(), "st_file_attributes", 0) & (0x400 | 0x1000 | 0x40000 | 0x400000)):
            raise ValueError("Linked or offline output file rejected")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in payload:
            info = zipfile.ZipInfo(PREFIX + name, date_time=(2026, 11, 9, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, data)
    members = []
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None and z.namelist() == [PREFIX + n for n in allow]
        for name, original in payload:
            data = z.read(PREFIX + name)
            assert data == original
            assert not findings(name, data, terms)
            members.append({"path": PREFIX + name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    tree = publishing_tree(dest, payload)
    manifest = {"archive": {"path": "dist/" + output.name, "bytes": output.stat().st_size,
                             "sha256": hashlib.sha256(output.read_bytes()).hexdigest()},
                "tree": tree, "members": members, "payload_bytes": sum(x["bytes"] for x in members),
                "verification": {"exact_allowlist": True, "CRC_and_member_hashes": True,
                                 "actual_archive_text_scanned": True, "png_preview_metadata_free": True,
                                 "generic_patterns": len(PATTERNS),
                                 "known_private_terms_checked": len(terms), "findings": [],
                                 "no_raw_datasets_weights_internal_reports_or_notebooks": True,
                                 "reviewed_dataset_derived_preview_and_metadata_only": True,
                                 "path_specific_metadata_size_limits": MEMBER_SIZE_LIMITS,
                                 "runtime_execution_checked_by_exporter": False,
                                 "runtime_qualification_evidence_is_separate": True}}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-terms", type=Path, help="Optional local owner identifier inventory; never exported")
    args = parser.parse_args()
    result = build(args.private_terms)
    print(json.dumps({"archive": result["archive"], "members": len(result["members"]), "verification": result["verification"]}, indent=2))
