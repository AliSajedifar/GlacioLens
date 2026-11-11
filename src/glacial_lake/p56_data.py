"""TRAIN-only source-balanced pools and stateless original-grid supplementary geometry."""
import collections,random
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from .p17_data import NativeDEMDataset
from .p03_data import apply_spatial
from .p56_native_data import ComponentDataset,collate_regions
from .common import digest

ROLES=('TINY','SMALL','COMPLETE','EDGE')
STRUCTURE=np.ones((3,3),bool)

def components(target,valid):
    labels,n=ndimage.label((target==1)&valid,structure=STRUCTURE)
    areas=np.bincount(labels.ravel(),minlength=n+1)
    invalid_neighbor=ndimage.binary_dilation(~valid,structure=STRUCTURE,border_value=0)
    invalid_ids=set(np.unique(labels[invalid_neighbor]).tolist())
    records=[]
    for cid,sl in enumerate(ndimage.find_objects(labels),1):
        if sl is None:continue
        y,x=sl;bbox=[y.start,y.stop,x.start,x.stop]
        edge=y.start==0 or y.stop==256 or x.start==0 or x.stop==256
        invalid=cid in invalid_ids
        records.append(dict(component_id=cid,area=int(areas[cid]),bbox=bbox,edge=edge,invalid_contact=invalid,complete=not edge and not invalid))
    return records

def legal_origins(bbox):
    y0,y1,x0,x1=bbox
    return [(max(0,min(256,b)-128),min(128,max(0,a-16))) for a,b in ((y0,y1+16),(x0,x1+16))]

def build_pools(root,rows,progress=None):
    from .data import read_tile
    pools={r:{str(c):{} for c in ((0,) if r in ROLES[:2] else (0,1))} for r in ROLES}
    excluded=collections.Counter();all_counts=collections.Counter();areas={r:[] for r in ROLES}
    for i,row in enumerate(rows):
        _,target,valid,info=read_tile(root,row)
        assert info['valid_pixels']==row['valid_pixels'] and info['positive_pixels']==row['positive_pixels']
        for a in components(target,valid):
            area=a['area'];role=None;cell=0
            all_counts['all_components']+=1;all_counts['invalid_contact']+=a['invalid_contact']
            if a['complete'] and 16<=area<256:
                role='TINY' if area<64 else 'SMALL'
                if any(lo>hi for lo,hi in legal_origins(a['bbox'])):
                    excluded[role]+=1;continue
            elif area>=256 and not a['invalid_contact']:
                role='EDGE' if a['edge'] else 'COMPLETE';cell=int(area>=4096)
            if role:
                pools[role][str(cell)].setdefault(row['id'],[]).append(a);areas[role].append(area)
        if progress and (i+1)%500==0:progress(i+1)
    for role in ROLES:
        assert any(pools[role].values()),'Real TRAIN role is empty: '+role
        for cell,images in pools[role].items():
            pools[role][cell]={sid:sorted(images[sid],key=lambda a:a['component_id']) for sid in sorted(images)}
    summary=dict(cells={r:{c:dict(images=len(v),objects=sum(map(len,v.values()))) for c,v in cells.items()} for r,cells in pools.items()},fit_window_exclusions=dict(excluded),all_component_counts=dict(all_counts),anchor_area={r:dict(min=min(v),median=float(np.median(v)),max=max(v)) for r,v in areas.items()},train_images=len(rows),ordered_train_ids_sha256=digest([r['id'] for r in rows]),pool_sha256=digest(pools))
    return pools,summary

def slot(pools,e,u,role,cell):
    realized=cell
    if not pools[role][str(realized)]:
        assert role in ('COMPLETE','EDGE');realized=1-cell
    images=pools[role][str(realized)];assert images
    sid=random.Random(f'p53-source-v1-{e}-{u}-{role}-{cell}').choice(sorted(images))
    a=random.Random(f'p53-anchor-v1-{e}-{u}-{role}-{cell}-{sid}').choice(images[sid])
    rng=random.Random(f'p53-d4-v1-{e}-{u}-{role}-{cell}-{sid}')
    result=dict(epoch=e,update=u,role=role,requested_cell=cell,realized_cell=realized,source_id=sid,anchor=a,turns=rng.randrange(4),flip=rng.random()<.5)
    if role in ('TINY','SMALL'):
        rng=random.Random(f'p53-crop-v1-{e}-{u}-{role}-{cell}-{sid}-{a["component_id"]}')
        result['crop_origin']=[rng.randint(lo,hi) for lo,hi in legal_origins(a['bbox'])]
    return result

def build_plan(pools):
    plan=[];counts=collections.Counter();fallback=collections.Counter();cells=collections.Counter()
    for e in range(24):
        epoch=[]
        for u in range(957):
            pairs=[]
            if u<956 or e%2==0:pairs.append([slot(pools,e,u,'TINY',0),slot(pools,e,u,'SMALL',0)])
            if u<956 or e%2==1:
                c=(e*956+u)%2 if u<956 else (e//2)%2
                pairs.append([slot(pools,e,u,'COMPLETE',c),slot(pools,e,u,'EDGE',1-c)])
            for pair in pairs:
                for s in pair:
                    counts[s['role']]+=1;cells[s['role']+':'+str(s['requested_cell'])]+=1
                    if s['requested_cell']!=s['realized_cell']:fallback[s['role']+':'+str(s['requested_cell'])]+=1
            epoch.append(pairs)
        plan.append(epoch)
    assert set(counts.values())=={22956} and sum(counts.values())==91824
    assert all(cells[r+':'+str(c)]==11478 for r in ('COMPLETE','EDGE') for c in (0,1))
    return plan,dict(roles=dict(counts),requested_cells=dict(cells),fallback_counts=dict(fallback),plan_sha256=digest(plan),total_aux_views=91824)

class AuxiliaryDataset(NativeDEMDataset):
    def __init__(self,root,rows,stats,*,dem_root):
        super().__init__(root,rows,stats,'B',augment=False,dem_root=dem_root);self.positions={r['id']:i for i,r in enumerate(rows)}
    def __getitem__(self,s):
        r=super().__getitem__(self.positions[s['source_id']]);record=self.sidecars[r['id']]
        with np.load(self.dem_root/'chips'/record['member']) as z:usable=z['valid']&np.isfinite(z['metres'])
        is_zoom=s['role'] in ('TINY','SMALL')
        y,x=s.get('crop_origin',(0,0));length=128 if is_zoom else 256
        def transform(a):
            a=a[...,y:y+length,x:x+length]
            a=np.rot90(a,s['turns'],axes=(-2,-1))
            if s['flip']:a=np.flip(a,axis=-1)
            return torch.from_numpy(a.copy())
        r['inputs']={k:transform(v.numpy()) for k,v in r['inputs'].items()}
        r['target']=transform(r['target'].numpy());r['valid']=transform(r['valid'].numpy());usable=transform(usable)
        if is_zoom:
            r['inputs']={k:F.interpolate(v[None].float(),size=(256,256),mode='bilinear',align_corners=False)[0] for k,v in r['inputs'].items()}
            usable=F.interpolate(usable[None,None].float(),size=(256,256),mode='nearest')[0,0].bool()
        r['inputs']['DEM'][:,~usable]=0
        assert all(v.dtype==torch.float32 and v.shape[-2:]==(256,256) and torch.isfinite(v).all() for v in r['inputs'].values())
        r.pop('dem_source_valid');r['dem_usable']=usable;r['is_zoom']=is_zoom
        return r

def verify_real_slots(dataset,plan):
    # Actual planned geometry only; no model or fitting. Cover both area cells and roles.
    slots=[s for pairs in plan[0][:2] for pair in pairs for s in pair]
    records=[]
    for s in slots:
        raw=NativeDEMDataset.__getitem__(dataset,dataset.positions[s['source_id']]);r=dataset[s]
        y,x=s.get('crop_origin',(0,0));size=128 if r['is_zoom'] else 256
        a=s['anchor'];y0,y1,x0,x1=a['bbox']
        assert y<=y0<y1<=y+size and x<=x0<x1<=x+size
        if r['is_zoom']:
            assert y<=max(0,y0-16) and y+size>=min(256,y1+16)
            assert x<=max(0,x0-16) and x+size>=min(256,x1+16)
        expected=raw['valid'][y:y+size,x:x+size]
        expected_positive=(raw['target'][y:y+size,x:x+size].bool()&expected).sum()
        assert r['valid'].sum()==expected.sum() and (r['target'].bool()&r['valid']).sum()==expected_positive
        assert (r['inputs']['DEM'][:,~r['dem_usable']]==0).all()
        records.append(dict(slot=s,target_shape=list(r['target'].shape),valid_pixels=int(r['valid'].sum()),positive_pixels=int(expected_positive),usable_DEM_pixels=int(r['dem_usable'].sum()),anchor_and_available_context_contained=True,dense_support_exact=True,unusable_DEM_zero=True))
    return dict(real_planned_slots=records,model_calls=0,global_rng_untouched_by_planning=True)
