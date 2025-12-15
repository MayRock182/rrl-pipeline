import numpy as np
from scipy.interpolate import interp1d
from utils import center_mask

def stack_rrls(rrl_list, rrl_window_size=2000, rrl_mask_width=100, weighted = False, weights=None):
    """
    Stacks a set of rrl spectra. Uses velocity space for interpolation

    Return the stacked spectrum.
    """
    chan_size = np.max(np.abs(np.diff(spec[:,1]))) # figure it out
    interp_chan = np.arange(-rrl_window_size/2., rrl_window_size/2., chan_size)
    stack_flx = np.zeros(len(flux_interp), weighted)
    signal_stacks = []
    rms_curve = []
    weights = []

    for idx, rrl_window in enumerate(rrl_list):
        interp_func = interp1d(rrl_window[:, 1], rrl_window[:, 2], bounds_error=False)
        flux_interp = interp_func(interp_chan)
        weights.append(weights)
        mask = center_mask()

        if weighted:
            flux_interp *= weights[idx]
            divisor = np.sum(weights)
        else:
            divisor = len(weights) # figure it out
        
        stack_flx += flux_interp
        rms_curve.append(np.std(stack_flx[mask]))
        signal_stacks.append(min((stack_flx/divisor)[~mask]))

    final_stack = stack_flx/divisor

    return interp_chan, final_stack, rms_curve, signal_stacks
