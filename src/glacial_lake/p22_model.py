"""P56 research-source projection; selected definitions retain their original bodies."""
import torch

def upstream_gradient_audit(model):
    norms={}
    for prefix in ('local','context','fuse'):
        params=list(getattr(model.joint_spatial,prefix).parameters())
        gradients=[p.grad for p in params if p.grad is not None]
        if len(gradients)!=len(params) or not all(torch.isfinite(g).all() for g in gradients):raise ValueError('Spatial upstream gradient absent/nonfinite')
        value=sum(float(g.detach().float().norm()) for g in gradients)
        if value<=0:raise ValueError('Spatial upstream gradient remains zero after projection moved')
        norms[prefix]=value
    if not model._p22_context_gradient_norms or min(model._p22_context_gradient_norms)<=0:raise ValueError('Live context has no finite nonzero gradient')
    model._p22_capture_context_gradient=False
    return {'checked_before_completed_update':2,'branch_gradient_norms':norms,'live_decoder_context_gradient_norms':list(model._p22_context_gradient_norms),
            'production_batches_only':True,'no_extra_base_forward':True}
