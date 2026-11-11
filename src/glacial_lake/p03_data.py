"""P56 research-source projection; selected definitions retain their original bodies."""
import random
import numpy as np

def transform_description(epoch, index, seed=42):
    rng = random.Random(f'glb-{seed}-{epoch}-{index}')
    return rng.randrange(4), rng.random() < .5


def apply_spatial(array, description):
    turns, flip = description
    result = np.rot90(array, turns, axes=(-2, -1))
    return np.flip(result, axis=-1) if flip else result
