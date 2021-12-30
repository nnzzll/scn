import numpy as np


def Dice(pred: np.ndarray, target: np.ndarray, smooth: int = 1) -> float:
    pred = np.atleast_1d(pred.astype(np.bool))
    target = np.atleast_1d(target.astype(np.bool))

    intersction = np.count_nonzero(pred & target)

    dice_coef = (2.*intersction+smooth)/float(
        np.count_nonzero(pred)+np.count_nonzero(target)+smooth
    )
    return dice_coef