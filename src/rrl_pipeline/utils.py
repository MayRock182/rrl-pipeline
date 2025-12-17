import numpy as np

def center_mask(x, mask_width):
    return ((x < 0 * x.unit - mask_width/2) | (x > 0 * x.unit + mask_width/2)) & (x != 0)

def rchi2(res, order):
    # Testing MAD-based noise estimate
    dof = len(res) - order + 1
    median = np.nanmedian(res)
    mad = np.nanmedian(abs(res - median))
    sigma_hat = 1.4826 * mad
    red_chi2 = np.nansum((res/sigma_hat)**2)/max(dof, 1)
    
    return red_chi2

# Depricated due to residuals essentially canceling out in chi square calculation
def rchi(res, order):
    """
    Depricated. Use utils.rchi2() instead
    
    :param res: Description
    :param order: Description
    """
    sigma = res[~res.mask].std()
    chi2 = np.ma.sum((res/sigma)**2)
    
    return chi2/dof if dof > 0 else np.inf