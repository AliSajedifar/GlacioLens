"""P56 research-source projection; selected definitions retain their original bodies."""
import torch
from .p17_model import FirstUpdateAudit as NativeAudit

class FirstUpdateAudit(NativeAudit):
 def __init__(self,model,arm):super().__init__(model,'B')
 def finish(self,batch):
  result=super().finish(batch);p=self.model.joint_spatial.predict.weight
  assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.float().norm()>0
  result['spatial_path']=dict(self.model._p22_audit,first_projection_gradient_norm=float(p.grad.float().norm()),upstream_zero_initially_expected=True)
  result['positive_regions_per_image']=[len(r) for r in batch['regions']];return result
