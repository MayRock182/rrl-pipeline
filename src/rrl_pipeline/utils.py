import numpy as np

def center_mask(x, mask_width):
    return (x < len(x)/2-mask_width | x > len(x)/2+mask_width) & x != 0

def rchi(res, dof, sigma):
    sigma = np.std(res)

    return (res/sigma)**2/(len(res) - dof + 1)