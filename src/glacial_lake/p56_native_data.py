"""Exact original native-DEM inputs; explicit common seed234 geometry only."""
import torch
from torch.utils.data import default_collate
from .p17_data import NativeDEMDataset
from .p03_data import transform_description,apply_spatial
from .p23_loss import make_regions
class ComponentDataset(NativeDEMDataset):
 def __init__(self,root,rows,statistics,*,dem_root):super().__init__(root,rows,statistics,'B',augment=False,dem_root=dem_root)
 def __getitem__(self,index):
  if not isinstance(index,tuple):raise ValueError('P56 requires explicit epoch/item geometry')
  epoch,item=index;r=super().__getitem__(item);transform=transform_description(epoch,item,seed=234)
  for k,v in r['inputs'].items():r['inputs'][k]=torch.from_numpy(apply_spatial(v.numpy(),transform).copy())
  for k in ('target','valid','dem_source_valid'):r[k]=torch.from_numpy(apply_spatial(r[k].numpy(),transform).copy())
  r['regions']=make_regions(r['target'].numpy(),r['valid'].numpy());return r
def collate_regions(items):
 regions=[r.pop('regions') for r in items];batch=default_collate(items);batch['regions']=regions;return batch
