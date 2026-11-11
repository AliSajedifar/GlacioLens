"""P56 research-source projection; selected definitions retain their original bodies."""
import random
import numpy as np
import torch

def seed_training():
    random.seed(234);np.random.seed(234);torch.manual_seed(234);torch.cuda.manual_seed_all(234)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state_all(r['cuda'])
