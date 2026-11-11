"""Verify released ordered populations, file identities and the exact auxiliary schedule on CPU."""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def load_sampling(root, train_ids):
    """Expand losslessly indexed metadata into the original pool and nested epoch plan."""
    raw = json.loads((root / "auxiliary_pool.json").read_text(encoding="utf-8"))
    schedule = json.loads((root / "auxiliary_plan.json").read_text(encoding="utf-8"))
    roles = raw["roles"]
    pools = {role: {cell: {} for cell in raw["cells"][role]} for role in roles}
    objects = []
    for role_index, cell, train_index, component_id, area, bbox, complete, edge, invalid_contact in raw["objects"]:
        role, source_id = roles[role_index], train_ids[train_index]
        anchor = dict(component_id=component_id, area=area, bbox=bbox, complete=complete, edge=edge, invalid_contact=invalid_contact)
        pools[role][str(cell)].setdefault(source_id, []).append(anchor)
        objects.append((role, cell, source_id, anchor))
    plan = []
    for epoch, updates in enumerate(schedule["epochs"]):
        epoch_plan = []
        for update, groups in enumerate(updates):
            update_plan = []
            for records in groups:
                group = []
                for record in records:
                    object_index, requested_cell, flip, turns, *crop = record
                    role, realized_cell, source_id, anchor = objects[object_index]
                    row = dict(anchor=anchor, epoch=epoch, update=update, source_id=source_id,
                               role=role, requested_cell=requested_cell, realized_cell=realized_cell,
                               flip=flip, turns=turns)
                    if crop:
                        row["crop_origin"] = crop[0]
                    group.append(row)
                update_plan.append(group)
            epoch_plan.append(update_plan)
        plan.append(epoch_plan)
    return pools, plan


def verify(root, data_root=None):
    release = json.loads((root / "release.json").read_text(encoding="utf-8"))
    for name, record in release["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe metadata path")
        content = (root / name).read_bytes()
        if len(content) != record["bytes"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError("File identity mismatch: " + name)
    cohorts = {}
    for split, record in release["cohorts"].items():
        ids = json.loads((root / record["file"]).read_text(encoding="utf-8"))
        if len(ids) != record["count"] or len(set(ids)) != len(ids) or digest(ids) != record["ordered_ids_logical_sha256"]:
            raise ValueError("Ordered cohort identity mismatch: " + split)
        cohorts[split] = ids
    with (root / "source_members.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = [(split, str(i), name) for split, ids in cohorts.items() for i, name in enumerate(ids)]
    if [(r["split"], r["ordered_index"], r["canonical_id"]) for r in rows] != expected:
        raise ValueError("Source mapping does not follow the exact cohort order")
    checked = 0
    for row in rows:
        for kind in ("image", "label"):
            name = row["archive_relative_" + kind + "_path"]
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe source member path")
            if data_root is not None:
                path = data_root.joinpath(*relative.parts)
                if not path.is_file():
                    raise ValueError("Missing source member: " + name)
                expected_sha = row[kind + "_sha256"]
                if expected_sha:
                    hasher = hashlib.sha256()
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            hasher.update(chunk)
                    if hasher.hexdigest() != expected_sha:
                        raise ValueError("Raw source byte mismatch: " + name)
                    checked += 1
    pools, plan = load_sampling(root, cohorts["train"])
    aux = release["auxiliary"]
    if digest(pools) != aux["pool_logical_sha256"] or digest(plan) != aux["plan_logical_sha256"]:
        raise ValueError("Auxiliary pool/plan identity mismatch")
    views = sum(len(group) for epoch in plan for update in epoch for group in update)
    if len(plan) != 24 or views != 91824:
        raise ValueError("Auxiliary coverage mismatch")
    return {"cohorts": {k: len(v) for k, v in cohorts.items()}, "source_rows": len(rows),
            "file_hashes": len(release["files"]), "ordered_logical_digests": "exact",
            "auxiliary_epochs": len(plan), "auxiliary_views": views, "pool_and_plan_digests": "exact",
            "local_raw_files_byte_verified": checked, "raw_source_presence_checked": data_root is not None,
            "raw_dataset_complete_byte_verification": data_root is not None and checked == len(rows) * 2,
            "model_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/manifests")
    parser.add_argument("--data-root", type=Path, help="Optional extracted dataset parent; checks every mapped file and every available raw-file SHA256")
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest_dir, args.data_root), indent=2))
