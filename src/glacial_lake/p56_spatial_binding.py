"""Unchanged P22 forward binding; isolated fresh spatial initialization explicit task seed."""
import hashlib,inspect,types,torch
from .p22_spatial import JointSpatialPath

def install_spatial_path(model,initial,seed=234):
    before_cpu=torch.get_rng_state().clone()
    devices=list(range(torch.cuda.device_count()))
    before_cuda=torch.cuda.get_rng_state_all()
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        model.joint_spatial=JointSpatialPath()
    if not torch.equal(before_cpu,torch.get_rng_state()) or any(not torch.equal(a,b) for a,b in zip(before_cuda,torch.cuda.get_rng_state_all())):
        raise ValueError('Branch construction changed the historical RNG stream')
    assert sum(p.numel() for p in model.joint_spatial.parameters())==57249
    original=model.forward
    model._p22_audit={'branch_rng_restored':True,'added_parameters':57249,'extra_base_forwards':0,
        'decoder_class':type(model.decoder).__module__+'.'+type(model.decoder).__name__,
        'decoder_forward_source_sha256':hashlib.sha256(inspect.getsource(type(model.decoder).forward).encode()).hexdigest(),
        'original_forward_source_sha256':hashlib.sha256(inspect.getsource(type(model).forward).encode()).hexdigest()}
    model._p22_initial_pending=initial
    model._p22_capture_context_gradient=False
    model._p22_context_gradient_norms=[]
    def forward(self,inputs,**kwargs):
        captured=[]
        hook=self.decoder.register_forward_hook(lambda module,args,output:captured.append(output))
        try:result=original(inputs,**kwargs)
        finally:hook.remove()
        if len(captured)!=1 or not isinstance(captured[0],torch.Tensor):raise ValueError('Exactly one live decoder output required')
        context=captured.pop();base=result.output
        x=torch.cat([inputs[k] for k in ('S2L2A','S1GRD','DEM')],dim=1)
        base4=base[:,None] if base.ndim==3 else base
        if x.shape!=(base.shape[0],9,256,256) or context.shape!=(base.shape[0],256,64,64) or base4.shape!=(base.shape[0],1,256,256):
            raise ValueError('Actual P22 input/context/logit geometry differs')
        if self.training and torch.is_grad_enabled() and not context.requires_grad:raise ValueError('Live decoder context was detached')
        if self._p22_capture_context_gradient and context.requires_grad:
            context.register_hook(lambda g:self._p22_context_gradient_norms.append(float(g.detach().float().norm())))
        combined=self.joint_spatial(x,context,base4)
        if self._p22_initial_pending:
            if not torch.equal(combined,base4):raise ValueError('Zero spatial residual changed canonical B start')
            self._p22_audit.update(input_shape=list(x.shape),context_shape=list(context.shape),logits_shape=list(base4.shape),
                initial_zero_residual_exact=True,live_context_requires_grad=context.requires_grad,single_base_forward=True,
                additional_normalization=False,first_joint_input_sha256=hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest())
            self._p22_initial_pending=False
        return type(result)(output=combined[:,0] if base.ndim==3 else combined,auxiliary_heads=result.auxiliary_heads)
    model.forward=types.MethodType(forward,model)
    return model
