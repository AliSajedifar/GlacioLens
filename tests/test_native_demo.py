"""Exercise example extraction boundaries without a network or model."""
import hashlib
import importlib.util
import io
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location('native_demo', Path(__file__).resolve().parents[1] / 'tools/native_demo.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def fixture(names=('inputs/a.tif',), symlink=False):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for name in names:
            info = zipfile.ZipInfo(name)
            if symlink:
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, b'fixture')
    data = buffer.getvalue()
    return data, {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                  'files': {n: {'bytes': 7, 'sha256': hashlib.sha256(b'fixture').hexdigest()} for n in names}}


class NativeDemoDelivery(unittest.TestCase):
    def test_extract_and_refuse_existing_data(self):
        data, pin = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'demo'
            demo.extract(data, pin, destination)
            self.assertEqual((destination / 'inputs/a.tif').read_bytes(), b'fixture')
            with self.assertRaises(FileExistsError):
                demo.extract(data, pin, destination)
            self.assertEqual((destination / 'inputs/a.tif').read_bytes(), b'fixture')

    def test_corrupted_archive_rejected_before_destination_creation(self):
        data, pin = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'demo'
            with self.assertRaisesRegex(ValueError, 'archive identity'):
                demo.extract(data + b'x', pin, destination)
            self.assertFalse(destination.exists())

    def test_unsafe_and_duplicate_members_rejected(self):
        for names in [('../escape',), ('/absolute',), ('a\\b',), ('C:/a',), ('a', 'A')]:
            with self.subTest(names=names):
                with self.assertRaises(ValueError):
                    demo.verified_members(*fixture(names))

    def test_symlink_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            demo.verified_members(*fixture(symlink=True))

    def test_changed_member_hash_rejected(self):
        data, pin = fixture()
        pin['files']['inputs/a.tif']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'member identity'):
            demo.verified_members(data, pin)


if __name__ == '__main__':
    unittest.main()
