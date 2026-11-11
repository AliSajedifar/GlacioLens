"""Check generic local input routing without reading imagery or loading a model."""
import json
from pathlib import Path, PurePosixPath


def check_manifest(path, root=None, check_files=False):
    path = Path(path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a nonempty list")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Each row needs a nonempty string id")
        if row["id"] in seen:
            raise ValueError("Duplicate sample id")
        seen.add(row["id"])
        for key in ("image", "dem", "usability"):
            value = row.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError("Each row needs image, dem and usability paths")
            rel = PurePosixPath(value)
            if rel.is_absolute() or ".." in rel.parts or ":" in value or "\\" in value:
                raise ValueError("Use relative forward-slash asset paths without traversal")
            if check_files:
                base = Path(root or path.parent).resolve()
                target = base.joinpath(*rel.parts).resolve()
                if not target.is_relative_to(base) or not target.is_file():
                    raise ValueError("Missing asset or asset outside supplied root")
    return {"rows": len(rows), "routing_schema_valid": True,
            "files_checked": check_files, "raster_geometry_checked": False, "model_calls": 0}
