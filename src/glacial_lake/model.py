"""Pinned TerraTorch adapter with real checkpoint, feature, and gradient audits."""
from __future__ import annotations

import contextlib
import math
import types
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def group_norm_new_modules(module):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            groups = math.gcd(32,child.num_features)
            replacement = nn.GroupNorm(groups,child.num_features,eps=child.eps,affine=child.affine)
            if child.affine:
                with torch.no_grad():
                    replacement.weight.copy_(child.weight); replacement.bias.copy_(child.bias)
            setattr(module,name,replacement); count += 1
        else:
            count += group_norm_new_modules(child)
    return count


def activation_checkpoint_blocks(encoder):
    """Wrap existing bound methods, retaining parameter names and RNG behavior."""
    for block in encoder.encoder:
        if getattr(block,'_glb_checkpoint_enabled',False):
            continue
        original = block.forward
        def forward(self, *args, _forward=original, **kwargs):
            if self.training and torch.is_grad_enabled():
                return checkpoint(_forward,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
            return _forward(*args,**kwargs)
        block.forward = types.MethodType(forward,block)
        block._glb_checkpoint_enabled = True


@contextlib.contextmanager
def audited_backbone_load(audit, sentinels):
    """Scope instrumentation to the pinned backbone class during one build only."""
    from terratorch.models.backbones.terramind.model.terramind_vit import TerraMindViT
    original = TerraMindViT.load_state_dict
    def load(self, state_dict, *args, **kwargs):
        result = original(self,state_dict,*args,**kwargs)
        audit.update({'checkpoint_tensor_keys':len(state_dict),'missing_keys':list(result.missing_keys),
                      'unexpected_key_count':len(result.unexpected_keys),
                      'unexpected_key_examples':list(result.unexpected_keys)[:20]})
        if result.missing_keys:
            raise ValueError('Pretrained encoder has missing keys: '+repr(result.missing_keys[:20]))
        live = self.state_dict()
        for prefix in ('encoder.0.','encoder.23.'):
            key = next(k for k in live if k.startswith(prefix) and k.endswith('weight') and live[k].ndim>=2)
            sentinels[key] = state_dict[key].flatten()[:64].clone()
            if not torch.equal(live[key].flatten()[:64],sentinels[key]):
                raise ValueError('Pretrained tensor mismatch: '+key)
        for mod in self.encoder_embeddings:
            key = 'encoder_embeddings.'+mod+'.proj.weight'
            raw = state_dict[key]
            # Upstream flattens [pixel,band], not [band,pixel].
            if mod == 'untok_sen2l2a@224':
                bands, original_bands = [1,2,3,7,10,11], 12
            elif mod == 'untok_sen1grd@224':
                bands, original_bands = [0,1], 2
            elif mod == 'untok_dem@224':
                bands, original_bands = [0], 1
                for suffix in ('proj.weight', 'mod_emb', 'pos_emb'):
                    dem_key = 'encoder_embeddings.'+mod+'.'+suffix
                    if dem_key not in state_dict or not torch.equal(live[dem_key], state_dict[dem_key]):
                        raise ValueError('Native pretrained DEM tensor missing or unequal: '+dem_key)
                audit['native_dem'] = {'module': mod, 'projection_shape': list(raw.shape),
                    'entire_projection_and_embeddings_equal_public_checkpoint': True}
            else:
                raise ValueError('Unaudited input modality: '+mod)
            if raw.ndim != 2 or raw.shape[1] != 256*original_bands:
                raise ValueError('Unexpected pixel-major input projection: '+key)
            sentinels[key] = raw[:2].reshape(2,256,original_bands)[:,[0,127,255]][:,:,bands].clone()
        return result
    TerraMindViT.load_state_dict = load
    try:
        yield
    finally:
        TerraMindViT.load_state_dict = original


def build_model(config, checkpoint_path, resume=False):
    from terratorch.models.encoder_decoder_factory import EncoderDecoderFactory
    args = dict(config['model_args'])
    args['backbone_pretrained'] = False
    args['backbone_ckpt_path'] = None if resume else str(checkpoint_path)
    audit, sentinels = {}, {}
    with audited_backbone_load(audit,sentinels):
        model = EncoderDecoderFactory().build_model(task='segmentation',**args)
    if not resume:
        if not audit:
            raise RuntimeError('No observed local pretrained-load operation')
        live = model.encoder.state_dict()
        comparisons = {}
        for key, expected in sentinels.items():
            value = live[key]
            if 'encoder_embeddings.' in key:
                actual = value[:2].reshape(2,256,-1)[:,[0,127,255]]
            else:
                actual = value.flatten()[:64]
            comparisons[key] = bool(torch.equal(actual,expected))
        if not all(comparisons.values()):
            raise ValueError('Selected pretrained projection or encoder sentinel differs')
        audit['sentinel_comparisons'] = comparisons
        audit['verified'] = True
    else:
        audit = {'verified':False,'route':'strict Lightning checkpoint restore; original load audit retained in run evidence'}
    del sentinels
    converted = sum(group_norm_new_modules(getattr(model,name)) for name in ('neck','decoder','head'))
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if config['profile'].get('activation_checkpointing'):
        activation_checkpoint_blocks(model.encoder)
    audit.update({'new_batchnorm_converted':converted,'encoder_trainable':all(p.requires_grad for p in model.encoder.parameters()),
                  'total_parameters':sum(p.numel() for p in model.parameters()),
                  'encoder_parameters':sum(p.numel() for p in model.encoder.parameters())})
    return model,audit


def optimizer_groups(model, config):
    groups={}
    norm_ids={id(p) for m in model.modules() if isinstance(m,(nn.LayerNorm,nn.GroupNorm,nn.BatchNorm2d)) for p in m.parameters(recurse=False)}
    encoder_ids={id(p) for p in model.encoder.parameters()}
    explicit_no_decay = set(model.encoder.no_weight_decay()) if hasattr(model.encoder,'no_weight_decay') else set()
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise ValueError('Unexpected frozen parameter: '+name)
        lr = config['encoder_lr'] if id(parameter) in encoder_ids else config['new_module_lr']
        local_name = name.removeprefix('encoder.')
        decay = 0.0 if name.endswith('.bias') or parameter.ndim<=1 or id(parameter) in norm_ids or local_name in explicit_no_decay else config['weight_decay']
        groups.setdefault((lr,decay),[]).append(parameter)
    result=[{'params':params,'lr':lr,'weight_decay':decay} for (lr,decay),params in groups.items()]
    covered=[id(p) for g in result for p in g['params']]
    if len(covered)!=len(set(covered)) or set(covered)!={id(p) for p in model.parameters()}:
        raise ValueError('Optimizer parameter coverage mismatch')
    return result


class FeatureAudit:
    def __init__(self,model):
        self.shapes={}; self.handles=[]
        def encoder_hook(module, args, outputs):
            self.shapes['retained_features']=[list(outputs[i].shape) for i in (7,11,15,23)]
        def neck_hook(module,args,outputs):
            self.shapes['pyramid']=[list(x.shape) for x in outputs]
        self.handles=[model.encoder.register_forward_hook(encoder_hook),model.neck.register_forward_hook(neck_hook)]

    def close_and_validate(self):
        for handle in self.handles:
            handle.remove()
        self.handles=[]
        if not self.shapes:
            raise RuntimeError('No feature observation')
        b=self.shapes['retained_features'][0][0]
        if self.shapes['retained_features'] != [[b,256,1024]]*4:
            raise ValueError('Unexpected TerraMind feature shapes: '+repr(self.shapes))
        expected=[[b,256,64,64],[b,512,32,32],[b,1024,16,16],[b,1024,8,8]]
        if self.shapes['pyramid'] != expected:
            raise ValueError('Unexpected pyramid shapes: '+repr(self.shapes))
        return self.shapes


def gradient_audit(model):
    result={}
    for name,parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError('Nonfinite gradient: '+name)
    prefixes = tuple('encoder.encoder_embeddings.'+mod+'.proj.' for mod in model.encoder.encoder_embeddings)
    for prefix in prefixes + ('encoder.encoder.0.','encoder.encoder.23.','head.'):
        matches=[(n,p) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        if not matches:
            raise RuntimeError('Missing required gradient: '+prefix)
        name,p=matches[0]
        norm=float(p.grad.detach().float().norm())
        if norm == 0:
            raise RuntimeError('Zero representative gradient: '+name)
        result[name]={'finite':True,'norm':norm}
    return result
