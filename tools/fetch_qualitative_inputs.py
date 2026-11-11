"""Fetch only the three original RGB-source TIFFs needed for the public figure.

Uses bounded HTTP byte ranges on the public GLB ZIP; never downloads the whole
archive. Existing byte-identical files are reused. Dataset terms: CC BY 4.0.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


class BoundedRanges(io.RawIOBase):
    def __init__(self, url, size, budget=32_000_000):
        self.url, self.size, self.budget = url, size, budget
        self.pos = self.total = self.requests = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2): raise ValueError('Invalid seek mode')
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if not 0 <= self.pos <= self.size: raise ValueError('Seek outside source archive')
        return self.pos

    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        if not 0 <= n <= 8_000_000: raise ValueError('Unexpected oversized range request')
        n = min(n, self.size - self.pos)
        if not n: return b''
        if self.total + n > self.budget: raise ValueError('Range-download byte budget exhausted')
        start, end = self.pos, self.pos + n - 1
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(req, timeout=45) as response:
            if response.status != 206:
                raise ValueError('Server did not honor byte ranges; refusing full archive download')
            if response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise ValueError('Source archive/range identity differs')
            data = response.read(n + 1)
            if len(data) != n: raise ValueError('Unexpected range length')
        self.pos += n; self.total += n; self.requests += 1
        return data


def digest(data): return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root / 'docs/figures/glaciolens_qualitative.json')
    parser.add_argument('--output-root', type=Path, default=Path('example_data'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    out = args.output_root.resolve()
    required, files = [], []
    for case in config['cases']:
        path = (out / case['source_image_member']).resolve()
        if not path.is_relative_to(out): raise ValueError('Nonportable source member path')
        if path.is_file():
            if digest(path.read_bytes()) != case['image_sha256']:
                raise ValueError(f'Existing image differs: {case["id"]}; move it aside explicitly')
            files.append({'id':case['id'], 'source_member':case['source_image_member'],
                          'sha256':case['image_sha256'], 'reused':True})
        else:
            required.append((case, path))
    transferred = requests = 0
    if required:
        ranges = BoundedRanges(config['source_archive_url'], config['source_archive_bytes'])
        with ranges, zipfile.ZipFile(ranges) as archive:
            for case, path in required:
                raw = archive.read(case['source_image_member'])
                if digest(raw) != case['image_sha256']:
                    raise ValueError(f'Source image identity differs: {case["id"]}')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                files.append({'id':case['id'], 'source_member':case['source_image_member'],
                              'bytes':len(raw), 'sha256':case['image_sha256'], 'reused':False})
        transferred, requests = ranges.total, ranges.requests
    out.mkdir(parents=True, exist_ok=True)
    receipt = {'files':files, 'range_response_bytes':transferred, 'range_requests':requests,
               'whole_archive_downloaded':False, 'whole_archive_fresh_hash_verified':False,
               'source_url':config['source_archive_url'], 'dataset_license':'CC-BY-4.0'}
    (out / 'figure_input_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
