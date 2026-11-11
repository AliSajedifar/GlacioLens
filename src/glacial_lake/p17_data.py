"""Native metre DEM input; original optical/SAR normalization and label support."""
from pathlib import Path
import numpy as np
import torch
from .common import read_json
from .data import TileDataset
from .p03_data import transform_description, apply_spatial


class NativeDEMDataset(TileDataset):
    def __init__(self, root, rows, statistics, arm, augment=False, *, dem_root=None):
        super().__init__(root, rows, statistics, augment)
        self.arm = arm
        if arm not in ('A', 'B'):
            raise ValueError('Unknown P17 arm')
        if arm == 'B':
            if dem_root is None:
                raise ValueError('Supply the prepared physical DEM sidecar directory explicitly')
            self.dem_root = Path(dem_root)
            self.sidecars = {r['id']: r for r in read_json(self.dem_root/'sidecar_index.json')}
            if any(r['id'] not in self.sidecars for r in rows):
                raise ValueError('Missing original-grid DEM sidecar')

    def __getitem__(self, index):
        if self.augment and not isinstance(index, tuple):
            raise ValueError('P17 augmentation needs an explicit deterministic epoch/item seed')
        result = super().__getitem__(index)
        epoch, position = index if isinstance(index, tuple) else (None, index)
        original = self.rows[position]
        if (int(result['valid'].sum()) != original['valid_pixels'] or
                int((result['target'][result['valid']] == 1).sum()) != original['positive_pixels']):
            raise ValueError('Original target/valid support changed')
        if self.arm == 'B':
            row = self.sidecars[result['id']]
            with np.load(self.dem_root/'chips'/row['member'], allow_pickle=False) as data:
                metres, valid = data['metres'], data['valid']
                if str(data['grid_hash']) != row['grid_hash'] or metres.shape != (256,256) or valid.shape != metres.shape:
                    raise ValueError('DEM metre array/grid differs')
            if self.augment:
                transform = transform_description(epoch, position)
                metres, valid = [apply_spatial(a, transform).copy() for a in (metres, valid)]
            usable = valid & np.isfinite(metres)
            z = np.zeros((256,256), np.float32)
            z[usable] = ((metres[usable].astype(np.float64)-651.663)/928.168).astype(np.float32)
            result['inputs']['DEM'] = torch.from_numpy(z[None])
            result['dem_source_valid'] = torch.from_numpy(valid.copy())
            result['dem_imputed_pixels'] = int((~usable).sum())
        return result
