"""Valid-pixel BCE plus positive-sample Dice; pooled confusion metrics."""
from __future__ import annotations


def segmentation_loss(logits, target, valid, components=False):
    import torch
    import torch.nn.functional as F
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits[:,0]
    if logits.shape != target.shape or valid.shape != target.shape:
        raise ValueError('Logit, target and validity shapes differ')
    if not valid.any():
        return None
    if not torch.isfinite(logits[valid]).all():
        raise FloatingPointError('Nonfinite valid logits')
    bce = F.binary_cross_entropy_with_logits(logits[valid].float(),target[valid].float())
    probabilities = torch.sigmoid(torch.where(valid, logits.float(), 0))
    # torch.where prevents invalid NaNs from contaminating a masked reduction.
    probabilities = torch.where(valid, probabilities, 0)
    truth = torch.where(valid,target.float(),0)
    positives = truth.flatten(1).sum(1)
    include = positives > 0
    if include.any():
        intersection = (probabilities*truth).flatten(1).sum(1)
        denominator = probabilities.flatten(1).sum(1)+positives
        dice = (1-(2*intersection[include]+1e-6)/(denominator[include]+1e-6)).mean()
    else:
        dice = bce.new_zeros(())
    return (bce + dice, bce.detach(), dice.detach()) if components else bce + dice


def confusion_counts(prediction, target, valid):
    """Works with NumPy arrays or Torch tensors; prediction is already Boolean."""
    p, y, v = prediction, target > 0.5, valid
    return {'tp':int((p & y & v).sum()),'fp':int((p & ~y & v).sum()),
            'fn':int((~p & y & v).sum()),'tn':int((~p & ~y & v).sum())}


def pooled_metrics(counts):
    tp,fp,fn,tn = (counts[k] for k in ('tp','fp','fn','tn'))
    def ratio(n,d):
        return n/d if d else None
    lake, background = ratio(tp,tp+fp+fn),ratio(tn,tn+fp+fn)
    return {**counts,'valid_pixels':tp+fp+fn+tn,'lake_iou':lake,
            'lake_dice':ratio(2*tp,2*tp+fp+fn),'precision':ratio(tp,tp+fp),
            'recall':ratio(tp,tp+fn),'background_iou':background,
            'two_class_miou':(lake+background)/2 if lake is not None and background is not None else None,
            'undefined_convention':'null; two-class mIoU requires both class unions',
            'aggregation':'pixel-pooled confusion counts'}


def accumulation_groups(batches, count):
    """Yield actual accumulation groups; caller divides by actual valid count."""
    group=[]
    for batch in batches:
        group.append(batch)
        if len(group)==count:
            yield group; group=[]
    if group:
        yield group
