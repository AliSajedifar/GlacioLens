"""Prospective P56 selection; no model or numerical library imports."""
def improves(candidate, prior):
    return prior is None or candidate['metrics']['two_class_miou'] > prior['metrics']['two_class_miou']

def history_winner(history):
    eligible = [r for r in history if r['EMA_VAL_two_class_miou'] is not None]
    assert [r['epoch'] for r in eligible] == list(range(3, 25))
    return max(eligible, key=lambda r: r['EMA_VAL_two_class_miou'])
