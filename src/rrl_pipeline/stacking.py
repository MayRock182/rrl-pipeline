import numpy as np
from scipy.interpolate import interp1d
from rrl_pipeline.utils import center_mask

# TODO eventually need to find a way to I THINK weigh channels that are more often averaged than others

def stack_rrls(rrl_list, rrl_window_size=1000, rrl_mask_width=100, weighted = False, weights=None):
    """
    Stacks a set of rrl spectra. Uses velocity space for interpolation

    Return the stacked spectrum.
    """
    largest_chan = 0 
    for spec in rrl_list:
        chan = np.max(np.abs(np.diff(spec["Velocity"])))
        if chan > largest_chan:
            largest_chan = chan

    interp_chan = np.arange(-rrl_window_size/2., rrl_window_size/2., largest_chan)
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
            divisor = len(weights)
        
        stack_flx += flux_interp
        rms_curve.append(np.std(stack_flx[mask]))
        signal_stacks.append(min((stack_flx/divisor)[~mask]))

    final_stack = stack_flx/divisor

    return final_stack, rms_curve, signal_stacks
