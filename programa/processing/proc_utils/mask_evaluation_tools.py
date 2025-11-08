import numpy as np

def IoU(ref_mask, adj_mask):
        ref = (ref_mask == 255)
        adj = (adj_mask == 255)
        inter = np.logical_and(ref, adj).sum()
        union = np.logical_or(ref, adj).sum()
        return inter / union if union > 0 else 0.0