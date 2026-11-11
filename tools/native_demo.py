"""Download, preflight or compare the pinned three-case example; never run a model."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / 'examples/native_demo.json'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verified_members(data, pin):
    if len(data) != pin['bytes'] or sha256(data) != pin['sha256']:
        raise ValueError('Example archive identity mismatch')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        if len(names) != len(set(n.casefold() for n in names)) or set(names) != set(pin['files']):
            raise ValueError('Unexpected or duplicate example members')
        if sum(m.file_size for m in members) > 20_000_000:
            raise ValueError('Example decompressed size exceeds limit')
        result = {}
        for member in members:
            name = member.filename
            path = PurePosixPath(name)
            if (path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name
                    or not path.parts or path.as_posix() != name or member.is_dir()
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError('Unsafe example member')
            expected = pin['files'][name]
            if member.file_size != expected['bytes']:
                raise ValueError('Example member size mismatch')
            raw = archive.read(member)
            if sha256(raw) != expected['sha256']:
                raise ValueError('Example member identity mismatch')
            result[name] = raw
        return result


def extract(data, pin, destination):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    members = verified_members(data, pin)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination; no archive path or pre-existing data is overwritten.
    with tempfile.TemporaryDirectory(prefix='.native_demo-', dir=destination.parent) as temporary:
        staged = Path(temporary) / 'payload'
        staged.mkdir()
        for name, raw in members.items():
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(raw)
        destination.mkdir(exist_ok=False)
        for child in staged.iterdir():
            shutil.move(str(child), str(destination / child.name))
    return {'status': 'downloaded_and_verified', 'files': len(members), 'sha256': pin['sha256']}


def download(pin, destination):
    if Path(destination).exists() or Path(destination).is_symlink():
        raise FileExistsError(destination)
    revision = pin['revision']
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('An immutable example revision is required')
    expected = 'https://huggingface.co/MaxiWb/glaciolens-glb-v1/resolve/' + revision + '/examples/native_demo.zip'
    if pin['url'] != expected or not 0 < pin['bytes'] <= 20_000_000:
        raise ValueError('Unexpected example location or size')
    with urllib.request.urlopen(pin['url'], timeout=120) as response:
        data = response.read(pin['bytes'] + 1)
    return extract(data, pin, destination)


def check_files(root, pin):
    root = Path(root).resolve(strict=True)
    for name, expected in pin['files'].items():
        path = root / name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError('Example path escapes its directory')
        raw = path.read_bytes()
        if len(raw) != expected['bytes'] or sha256(raw) != expected['sha256']:
            raise ValueError('Example file identity mismatch: ' + name)
    return json.loads((root / 'qualified_reference.json').read_text(encoding='utf-8'))


def preflight(root, pin):
    from glacial_lake.native_inference import prepare_job
    root = Path(root).resolve(strict=True)
    reference = check_files(root, pin)
    with tempfile.TemporaryDirectory() as temporary:
        job = prepare_job(root / 'inputs.json', Path(temporary) / 'unused_output')
    if [r['id'] for r in job.rows] != reference['ordered_ids']:
        raise ValueError('Qualification order differs')
    for actual, expected in zip(job.inputs, reference['inputs']):
        for key in ('id', 'input_sha256', 'grid', 'grid_sha256', 'input_valid_pixels', 'normalized_sha256'):
            if actual[key] != expected[key]:
                raise ValueError('Qualified input differs: ' + key)
    return {'status': 'qualified_inputs_match', 'order': reference['ordered_ids'],
            'physical_batches': [2, 1], 'model_calls': 0}


def compare(root, output, pin):
    import numpy as np
    import rasterio
    from glacial_lake.common import digest
    reference = check_files(root, pin)
    comparisons = []
    for expected in reference['expected_outputs']:
        path = Path(output) / expected['file']
        with rasterio.open(path) as raster:
            pixels = raster.read(1)
            grid = dict(crs=raster.crs.to_wkt() if raster.crs else None,
                        transform=list(raster.transform)[:6], height=raster.height,
                        width=raster.width, bounds=list(raster.bounds))
            if (raster.count != 1 or raster.dtypes != ('uint8',) or raster.nodata != 255
                    or digest(grid) != expected['grid_sha256']):
                raise ValueError('Output geometry/encoding differs: ' + expected['id'])
            support = raster.read_masks(1) > 0
        with rasterio.open(Path(root) / 'expected' / expected['file']) as retained:
            target = retained.read(1)
            target_support = retained.read_masks(1) > 0
        differences = int(np.count_nonzero(pixels != target))
        if differences or not np.array_equal(support, target_support):
            raise ValueError('Output pixels/support differ: ' + expected['id'])
        comparisons.append({'id': expected['id'], 'differing_pixels': differences,
                            'lake_pixels': int((pixels == 1).sum()), 'geometry_and_support_match': True,
                            'container_bytes_match': sha256(path.read_bytes()) == expected['sha256']})
    return {'status': 'masks_match', 'cases': comparisons, 'model_calls': 0,
            'scope': 'Decoded output comparison only; execution order and device are recorded in the prediction receipt.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('download', 'preflight', 'compare'))
    parser.add_argument('--directory', type=Path, default=Path('examples/native_demo'))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    pin = json.loads(PIN.read_text(encoding='utf-8'))
    if args.action == 'download':
        result = download(pin, args.directory)
    elif args.action == 'preflight':
        result = preflight(args.directory, pin)
    else:
        if args.output_dir is None:
            parser.error('compare requires --output-dir')
        result = compare(args.directory, args.output_dir, pin)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
