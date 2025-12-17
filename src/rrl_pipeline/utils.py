import numpy as np

def center_mask(x, mask_width):
    return ((x < 0 * x.unit - mask_width/2) | (x > 0 * x.unit + mask_width/2)) & (x != 0)

def rchi(res, order):
    sigma = res.std()
    dof = len(res) - order + 1
    chi2 = np.ma.sum((res/sigma)**2)

    return chi2/dof if dof > 0 else np.inf