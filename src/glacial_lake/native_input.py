"""Target-free input contract for the separately identified P56-native-r1 API."""
from pathlib import Path
import json
import re

import numpy as np

from .common import digest, hash_file
from .data import read_registered_image


def read_manifest(path):
    """Resolve explicit local assets and preserve the caller's sequence exactly."""
    path = Path(path).resolve(strict=True)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a nonempty JSON list")
    result, seen = [], set()
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "image", "dem", "usability"}:
            raise ValueError("Each row must contain exactly id, image, dem, usability")
        ident = row["id"]
        if (not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", ident)
                or ident.endswith(".") or ident.split(".")[0].upper() in reserved or ident.casefold() in seen):
            raise ValueError("IDs must be unique, portable file stems")
        seen.add(ident.casefold())
        resolved = {"id": ident}
        for key in ("image", "dem", "usability"):
            value = row[key]
            if not isinstance(value, str) or not value or "://" in value or value.startswith("/vsi"):
                raise ValueError("Input paths must identify local files")
            asset = Path(value)
            if not asset.is_absolute():
                asset = path.parent / asset
            asset = asset.resolve(strict=True)
            if not asset.is_file():
                raise ValueError("Input asset is not a file")
            resolved[key] = asset
        result.append(resolved)
    return result


def read_input(row, statistics):
    """Read full-support, aligned native GLB assets; never open a label raster.

    The NumPy operations reproduce the retained P56 input transform. Incomplete
    support is deliberately rejected because that case is not qualified here.
    """
    import rasterio

    with rasterio.open(row["image"]) as image:
        x, valid = read_registered_image(image)
        if image.crs is None:
            raise ValueError("A georeferenced input is required")
        grid = dict(crs=image.crs.to_wkt(), transform=list(image.transform)[:6],
                    height=image.height, width=image.width, bounds=list(image.bounds))
        profile = dict(driver="GTiff", height=image.height, width=image.width,
                       crs=image.crs, transform=image.transform, count=1, dtype="uint8",
                       nodata=255, compress="deflate")
        arrays = {}
        for key in ("dem", "usability"):
            with rasterio.open(row[key]) as ds:
                if (ds.count != 1 or (ds.height, ds.width) != (image.height, image.width)
                        or ds.crs != image.crs or ds.transform != image.transform):
                    raise ValueError("DEM and usability must have the exact image grid")
                arrays[key] = ds.read(1)
                if not (ds.read_masks(1) > 0).all():
                    raise ValueError("Only full-support native inputs are qualified")
    metres, usability = arrays["dem"], arrays["usability"]
    if (not valid.all() or not np.isfinite(metres).all() or not np.isfinite(usability).all()
            or not np.all(usability == 1)):
        raise ValueError("Only full-support native inputs with usability=1 are qualified")
    mean = np.array(statistics["mean"], dtype=np.float32)[:, None, None]
    std = np.array(statistics["std"], dtype=np.float32)[:, None, None]
    normalized = np.where(valid[None], (x - mean) / std, 0).astype(np.float32)
    dem = np.zeros(metres.shape, dtype=np.float32)
    usable = (usability > 0) & np.isfinite(metres)
    dem[usable] = ((metres[usable].astype(np.float64) - 651.663) / 928.168).astype(np.float32)
    inputs = {"S2L2A": normalized[:6].copy(), "S1GRD": normalized[6:].copy(), "DEM": dem[None].copy()}
    if not all(np.isfinite(a).all() for a in inputs.values()):
        raise ValueError("Nonfinite normalized input")
    import hashlib
    receipt = dict(id=row["id"], input_sha256={k: hash_file(row[k]) for k in ("image", "dem", "usability")},
                   grid=grid, grid_sha256=digest(grid), input_valid_pixels=int(valid.sum()),
                   normalized_sha256={k: hashlib.sha256(a.tobytes()).hexdigest() for k, a in inputs.items()})
    return inputs, valid, profile, receipt
