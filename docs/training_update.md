# Recorded optimizer-objective excerpt

Exact excerpt from the accepted executed `scripts/p56_train.py`, SHA256 `8764abb71974282100d4b9cde867899eb03f5b074b26c20449a80375426843f0`. This is explanatory source, not an executable training entry point. `n` is 16 (normal) or 4 (tail); `plan[e][u]` contains two auxiliary pairs or one at the tail. `forward_pair` uses the original journal and BF16 forward; optimizer/EMA/schedule are described in [method.md](method.md).

```python
for a in range(0,n,2):
    pair=dict(inputs={k:v[a:a+2] for k,v in batch['inputs'].items()},target=batch['target'][a:a+2],valid=batch['valid'][a:a+2],id=batch['id'][a:a+2])
    z,y,v=forward_pair(model,pair,journal,dict(kind='TRAIN_NATIVE',epoch=e,update=u,optimizer_update=step+1))
    dense=segmentation_loss(z,y,v);assert dense is not None
    positive=component_auxiliary_loss(z,batch['regions'][a:a+2]);negative=negative_region_loss(z,make_negative_plans(z,y,v))
    loss=dense.float()+.20*positive+coefficient*negative;assert torch.isfinite(loss)
    (loss*(2/n)).backward()
    for k,value in [('native_dense',dense),('native_positive',positive),('native_negative',negative),('native_total',loss)]:values[k]+=float(value.detach())*(2/n)
    del z,y,v,dense,positive,negative,loss,pair
aux_values={}
for pair_index,slots in enumerate(plan[e][u]):
    aux=default_collate([aux_ds[slot] for slot in slots]);zoom=slots[0]['role']=='TINY'
    assert all(bool(x)==zoom for x in aux['is_zoom'])
    z,y,v=forward_pair(model,aux,journal,dict(kind='TRAIN_ZOOM' if zoom else 'TRAIN_EXTENT',epoch=e,update=u,optimizer_update=step+1,pair_index=pair_index,roles=[q['role'] for q in slots],requested_cells=[q['requested_cell'] for q in slots]))
    if zoom:z=F.avg_pool2d(z[:,None].float(),kernel_size=2,stride=2)[:,0]
    dense=segmentation_loss(z,y,v);assert dense is not None and torch.isfinite(dense)
    (dense*(.25/len(plan[e][u]))).backward();dense_value=float(dense.detach());values['aux_dense']+=dense_value/len(plan[e][u])
    aux_values['zoom' if zoom else 'extent']=dense_value
    del aux,z,y,v,dense
values['total']=values['native_total']+.25*values['aux_dense']
```
