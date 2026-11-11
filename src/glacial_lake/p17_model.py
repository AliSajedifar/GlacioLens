"""P56 research-source projection; selected definitions retain their original bodies."""
import hashlib
import torch
from .model import FeatureAudit, gradient_audit

def state_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FirstUpdateAudit:
    def __init__(self,model,arm):
        self.model,self.arm=model,arm;self.features=FeatureAudit(model);self.observed={}
        def joint(module,args):self.observed['joint_tokens_shape']=list(args[0].shape)
        self.handle=model.encoder.encoder[0].register_forward_pre_hook(joint)

    def finish(self,batch):
        self.handle.remove();shapes=self.features.close_and_validate()
        expected=768 if self.arm=='B' else 512
        if self.observed['joint_tokens_shape'] != [2,expected,1024]:raise ValueError('Actual joint token count differs')
        if set(batch['inputs'])!={'S2L2A','S1GRD',*(['DEM'] if self.arm=='B' else [])}:raise ValueError('Actual modalities differ')
        return {**self.observed,**shapes,'gradients':gradient_audit(self.model),
            'input_shapes':{k:list(v[:2].shape) for k,v in batch['inputs'].items()},
            'input_sha256':{k:hashlib.sha256(v[:2].numpy().tobytes()).hexdigest() for k,v in batch['inputs'].items()},
            'target_valid_sha256':{k:hashlib.sha256(batch[k][:2].numpy().tobytes()).hexdigest() for k in ('target','valid')},
            'first_ids':batch['id'][:2], 'total_parameters':sum(p.numel() for p in self.model.parameters()),
            'full_finetuning':all(p.requires_grad for p in self.model.parameters())}
