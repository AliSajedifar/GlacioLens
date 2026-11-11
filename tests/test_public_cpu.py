"""Public CPU regression checks; no model, checkpoint, Torch or GPU is needed."""
import csv
import hashlib
import importlib.resources
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from glacial_lake import cli, native_inference, native_input, saved_counts


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFIER = load_tool('verify_data_manifest')
OBJECTS = load_tool('object_boundary_metrics')


class PublicCpuContracts(unittest.TestCase):
    def test_released_metadata_and_sampling(self):
        result = VERIFIER.verify(ROOT / 'data/manifests')
        self.assertEqual(result['cohorts'], {'train': 15300, 'val': 1912, 'test': 1913, 'glbc': 1105})
        self.assertEqual(result['source_rows'], 20230)
        self.assertEqual(result['auxiliary_epochs'], 24)
        self.assertEqual(result['auxiliary_views'], 91824)
        self.assertEqual(result['model_calls'], 0)

    def test_metadata_byte_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'manifests'
            shutil.copytree(ROOT / 'data/manifests', root)
            with (root / 'test_ids.json').open('ab') as stream:
                stream.write(b' ')
            with self.assertRaisesRegex(ValueError, 'File identity mismatch'):
                VERIFIER.verify(root)

    def test_reordered_ids_are_rejected_after_file_hash_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'manifests'
            shutil.copytree(ROOT / 'data/manifests', root)
            filename = 'test_ids.json'
            path = root / filename
            ids = json.loads(path.read_text(encoding='utf-8'))
            ids[0], ids[1] = ids[1], ids[0]
            content = (json.dumps(ids) + '\n').encode('utf-8')
            path.write_bytes(content)
            release_path = root / 'release.json'
            release = json.loads(release_path.read_text(encoding='utf-8'))
            release['files'][filename] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
            release_path.write_text(json.dumps(release), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Ordered cohort identity mismatch'):
                VERIFIER.verify(root)

    def test_synthetic_recount_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            masks = Path(temporary) / 'masks.npz'
            subprocess.run([sys.executable, str(ROOT / 'examples/make_synthetic_masks.py'), str(masks)],
                           check=True, capture_output=True, text=True)
            process = subprocess.run([sys.executable, '-m', 'glacial_lake', 'recount', str(masks)],
                                     check=True, capture_output=True, text=True)
            result = json.loads(process.stdout)
            self.assertEqual(result['pooled'], [3, 2, 1, 6])
            self.assertEqual(result['metrics']['valid_pixels'], 12)
            self.assertEqual(result['metrics']['lake_iou'], 0.5)
            self.assertAlmostEqual(result['metrics']['two_class_miou'], 7 / 12)
            self.assertEqual(result['model_calls'], 0)

    def test_invalid_packed_masks_are_rejected(self):
        good = np.zeros((1, 1, 1), dtype=np.uint8)
        for bad in (np.zeros((1, 1, 2), dtype=np.uint8), good.astype(np.int64), good[0]):
            with self.subTest(shape=bad.shape, dtype=str(bad.dtype)):
                with self.assertRaises(ValueError):
                    saved_counts.packed_counts(bad, good, good)

    def test_strict_prediction_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('image.tif', 'dem.tif', 'usability.tif'):
                (root / name).touch()
            row = {'id': 'tile_a', 'image': 'image.tif', 'dem': 'dem.tif', 'usability': 'usability.tif'}
            manifest = root / 'inputs.json'
            malformed = [[], [dict(row, label='label.tif')], [row, dict(row, id='TILE_A')],
                         [dict(row, image='https://example.org/image.tif')], [dict(row, id='../tile')]]
            for rows in malformed:
                with self.subTest(rows=rows):
                    manifest.write_text(json.dumps(rows), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        native_input.read_manifest(manifest)

    def test_missing_input_fails_before_model_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / 'inputs.json'
            manifest.write_text(json.dumps([{'id': 'tile_a', 'image': 'missing.tif',
                                            'dem': 'dem.tif', 'usability': 'usability.tif'}]), encoding='utf-8')
            with patch.object(native_inference, 'Predictor', side_effect=AssertionError('Model construction was reached')) as constructor:
                with self.assertRaises(FileNotFoundError):
                    cli.main(['predict', '--checkpoint', str(root / 'unused.ckpt'), '--manifest', str(manifest),
                              '--output-dir', str(root / 'new_output')])
                constructor.assert_not_called()

    def test_existing_output_fails_before_raster_or_model_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('image.tif', 'dem.tif', 'usability.tif'):
                (root / name).touch()
            manifest = root / 'inputs.json'
            manifest.write_text(json.dumps([{'id': 'tile_a', 'image': 'image.tif',
                                            'dem': 'dem.tif', 'usability': 'usability.tif'}]), encoding='utf-8')
            output = root / 'existing_output'
            output.mkdir()
            with patch.object(native_inference, 'Predictor', side_effect=AssertionError('Model construction was reached')) as constructor:
                with patch.object(native_inference, 'read_input', side_effect=AssertionError('Raster reads were reached')):
                    with self.assertRaises(FileExistsError):
                        cli.main(['predict', '--checkpoint', str(root / 'unused.ckpt'), '--manifest', str(manifest),
                                  '--output-dir', str(output)])
                constructor.assert_not_called()

    def test_installed_inference_resources_match_source(self):
        package = importlib.resources.files('glacial_lake')
        for name in ('inference_config.json', 'inference_normalization.json'):
            self.assertEqual(package.joinpath(name).read_bytes(), (ROOT / 'src/glacial_lake' / name).read_bytes())
        configuration, statistics = native_inference._read_configuration()
        self.assertEqual(configuration['checkpoint']['sha256'],
                         '19bcdaa9f925da61ecc46aab755b265960c632aacdec582bd3050a6b84bd4230')
        self.assertEqual(len(statistics['mean']), 8)

    def test_reproduction_payload_hashes_and_groups(self):
        data = ROOT / 'results/reproduction'
        manifest = json.loads((data / 'source_manifest.json').read_text(encoding='utf-8'))
        for relative, expected in manifest['distributed_files'].items():
            path = (data / relative).resolve()
            self.assertEqual(path.parent, data.resolve())
            content = path.read_bytes()
            self.assertEqual(len(content), expected['bytes'])
            self.assertEqual(hashlib.sha256(content).hexdigest(), expected['sha256'])
        with (data / 'group_mapping.csv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1913)
        self.assertEqual(len({row['group_id'] for row in rows}), 1201)
        for row in rows:
            self.assertEqual(row['group_id'], row['tile_id'].rsplit('_', 1)[0])

    def test_object_matching_differs_from_coverage(self):
        target = np.zeros((20, 20), dtype=bool)
        target[4:8, 3:7] = True
        target[4:8, 10:14] = True
        prediction = np.zeros_like(target)
        prediction[4:8, 3:14] = True
        result = OBJECTS.tile_statistics(target, prediction, np.ones_like(target))
        self.assertEqual(result['counts'], {'tp': 32, 'fp': 12, 'fn': 0, 'tn': 356})
        self.assertEqual(result['matching']['0.25'], {'gt': 2, 'predicted': 1, 'tp': 1, 'fp': 0, 'fn': 1})
        self.assertEqual(result['matching']['0.5'], {'gt': 2, 'predicted': 1, 'tp': 0, 'fp': 1, 'fn': 2})
        self.assertEqual(result['size']['complete']['16-63']['coverage_ge50'], 2)

    def test_boundary_identity_and_empty_convention(self):
        target = np.zeros((20, 20), dtype=bool)
        target[4:8, 3:7] = True
        target[4:8, 10:14] = True
        valid = np.ones_like(target)
        exact = OBJECTS.finalize_statistics(OBJECTS.tile_statistics(target, target, valid))
        self.assertEqual(exact['boundary']['F1'], 1.0)
        self.assertEqual(exact['matching']['0.5']['F1'], 1.0)
        empty = OBJECTS.finalize_statistics(OBJECTS.tile_statistics(np.zeros_like(target), np.zeros_like(target), valid))
        self.assertIsNone(empty['boundary']['F1'])
        self.assertIsNone(empty['matching']['0.5']['F1'])
        self.assertEqual(empty['counts'], {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 400})

    @classmethod
    def tearDownClass(cls):
        if 'torch' in sys.modules:
            raise AssertionError('CPU tests unexpectedly imported Torch')


if __name__ == '__main__':
    unittest.main()
