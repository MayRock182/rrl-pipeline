import numpy as np

def center_mask(x, mask_width):
    return ((x < -mask_width/2) | (x > mask_width/2))

def rchi2(res, order):
    # Testing MAD-based noise estimate
    dof = len(res) - order + 1
    median = np.ma.median(res)
    mad = np.ma.median(abs(res - median))
    sigma_hat = 1.4826 * mad
    red_chi2 = np.ma.sum((res/sigma_hat)**2)/max(dof, 1)
    
    return red_chi2

# Depricated due to residuals essentially canceling out in chi square calculation
def rchi(residuals, order):
    """
    Depricated. Use utils.rchi2() instead
    
    :param res: Description
    :param order: Description
    """
    sigma = residuals[~residuals.mask].std()
    dof = len(residuals) - order + 1
    chi2 = np.ma.sum((residuals/sigma)**2)
    
    return chi2/dof if dof > 0 else np.inf

# Thinking of extending it to A, B, C, etc. for being able to satisfy different thresholds
def quality_factor(residual, zero_count_threshold = 0.1):
    """
    Gives A for good quality spectra for use in final stack. F for bad quality and will NOT
    be used in the final stack.
    
    :param res: Spectrum residuals
    :param zero_count_threshold: Percent of 0 (MALS flagged) points in spectrum residuals
    :param min_spec_threshold: Description
    """
    # Finding the number of 0s found in 
    residual_full = residual.unmasked
    mask = residual_full == 0
    count_0 = np.sum(len(residual_full[mask]))
    count_non_0 = np.sum(len(residual_full[~mask]))

    if (count_0 / count_non_0) > zero_count_threshold:
        print("F Assigned")
        return "F"
    else:
        return "A"