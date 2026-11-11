"""P56 research-source projection; selected definitions retain their original bodies."""
from pathlib import Path
import random
import numpy as np
from .common import safe_relative
CHANNELS = [0, 1, 2, 3, 4, 5, 9, 10]

def decode_target(array, valid):
    unique = set(np.unique(array[valid]).tolist())
    if unique <= {0,1}:
        encoding = '0/1'
        target = array.astype(np.float32)
    elif unique <= {0,255}:
        encoding = '0/255'
        target = (array == 255).astype(np.float32)
    else:
        raise ValueError('Unsupported observed target classes: ' + repr(sorted(unique)))
    return np.where(valid, target, 0).astype(np.float32), encoding


def read_registered_image(image):
    """Read the existing GLB image contract without requiring a target raster."""
    if image.count != 11 or (image.height, image.width) != (256, 256):
        raise ValueError('Expected a registered native 256x256, 11-band GLB tile')
    x = image.read(indexes=[i+1 for i in CHANNELS]).astype(np.float32)
    valid = (image.read_masks(indexes=[i+1 for i in CHANNELS]) > 0).all(0)
    valid &= np.isfinite(x).all(0)
    return x, valid


def normalized_inputs(x, valid, statistics):
    """The unchanged stored-domain train-statistic transform shared by inference."""
    import torch
    mean = np.array(statistics['mean'], dtype=np.float32)[:, None, None]
    std = np.array(statistics['std'], dtype=np.float32)[:, None, None]
    x = np.where(valid[None], (x-mean)/std, 0).astype(np.float32)
    return {'S2L2A': torch.from_numpy(x[:6].copy()),
            'S1GRD': torch.from_numpy(x[6:].copy())}


def read_tile(root, row):
    import rasterio
    image_path = Path(root).joinpath(*safe_relative(row['image']).parts)
    target_path = Path(root).joinpath(*safe_relative(row['target']).parts)
    with rasterio.open(image_path) as image, rasterio.open(target_path) as mask:
        if image.count != 11 or mask.count != 1:
            raise ValueError(f'Expected 11 image bands and one mask band: {row["id"]}')
        if (image.height,image.width) != (256,256) or (mask.height,mask.width) != (256,256):
            raise ValueError('Expected native 256x256 paired tiles: ' + row['id'])
        if image.crs != mask.crs or not image.transform.almost_equals(mask.transform):
            raise ValueError('Image/target geospatial alignment differs: ' + row['id'])
        x, valid = read_registered_image(image)
        y = mask.read(1)
        valid &= mask.read_masks(1) > 0
        valid &= np.isfinite(x).all(0) & np.isfinite(y)
        target, encoding = decode_target(y, valid)
        info = {'source_shape':[image.count,image.height,image.width], 'mask_encoding':encoding,
                'crs_present':image.crs is not None, 'image_nodata':str(image.nodata),
                'mask_nodata':str(mask.nodata), 'band_descriptions':list(image.descriptions),
                'valid_pixels':int(valid.sum()), 'positive_pixels':int((target[valid] == 1).sum())}
    return x, target, valid, info


class TileDataset:
    def __init__(self, root, rows, statistics, augment=False):
        self.root, self.rows, self.augment = root, rows, augment
        self.statistics = statistics
        self.mean = np.array(statistics['mean'],dtype=np.float32)[:,None,None]
        self.std = np.array(statistics['std'],dtype=np.float32)[:,None,None]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        augmentation_rng = random
        if isinstance(index, tuple):
            epoch, index = index
            augmentation_rng = random.Random(f'glb-42-{epoch}-{index}')
        row = self.rows[index]
        x,y,valid,_ = read_tile(self.root,row)
        if not self.augment:
            return {'inputs': normalized_inputs(x, valid, self.statistics),
                    'target': torch.from_numpy(y.copy()), 'valid': torch.from_numpy(valid.copy()), 'id': row['id']}
        x = np.where(valid[None],(x-self.mean)/self.std,0).astype(np.float32)
        if self.augment:
            k = augmentation_rng.randrange(4)
            x,y,valid = [np.rot90(a,k,axes=(-2,-1)) for a in (x,y,valid)]
            if augmentation_rng.random() < 0.5:
                x,y,valid = [np.flip(a,axis=-1) for a in (x,y,valid)]
        return {'inputs':{'S2L2A':torch.from_numpy(x[:6].copy()),'S1GRD':torch.from_numpy(x[6:].copy())},
                'target':torch.from_numpy(y.copy()),'valid':torch.from_numpy(valid.copy()),'id':row['id']}
