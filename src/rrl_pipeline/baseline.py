import numpy as np
from numpy.polynomial.polynomial import Polynomial
from utils import center_mask, rchi
from rrlpy import freq as frequency
from astropy import units as u

def subtrack_baseline(freq, flux, mask_width, order = 1):
        """
        Returns baseline-subtracted spectrum
        """
        mask = center_mask(freq, mask_width)
        freq = freq[mask]
        flux = flux[mask]
        
        pfit = Polynomial.fit(freq, flux, order)
        residuals = freq - pfit

        return residuals

def minimize_order_baseline(rrl_window, mask_width, max_order = 1):
        min_residuals = []
        min_rchi = 1000
        order = max_order
        freq = rrl_window[0]
        flux = rrl_window[2]

        for order in range(max_order + 1):
                rchi_i = rchi(residuals)
                residuals = subtrack_baseline(freq, flux, mask_width, max_order)

                if rchi_i < min_rchi:
                        min_residuals = residuals
                        min_order = order
                        min_rchi = rchi_i

        return min_residuals, min_order, min_rchi

def extract_rrl(freq, flux, rrl_window_size, species, z):
        species_str = "RRL_" + species

        n_array, f_array = frequency.find_lines_sb(freq, species_str, z=z)
        
        for n, f in zip(n_array, f_array):
                freq_to_vel = u.doppler_relativitstic(f*u.Hz)
                vel = freq.to(u.km / u.s, equivalencies = freq_to_vel)

                # Find the indices of the range ends.
                # This works bc the center 0 km/s are RRLs
                idx_low = np.argmin(abs(vel - rrl_window_size/2.))# argmin is used because they are shifting vel
                idx_hgh = np.argmin(abs(vel + rrl_window_size/2.))
                idx_low, idx_hgh = np.sort([idx_low, idx_hgh]) # certain that  idx low to high
                idx_hgh += 1 # Python excludes the upper end of the range.

                rrl_window_frq = freq[idx_low:idx_hgh]
                rrl_window_vel = vel[idx_low:idx_hgh]
                rrl_window_flx = flux[idx_low:idx_hgh]

        return (rrl_window_frq, rrl_window_vel, rrl_window_flx)
        
def extract_rrls(spec_list, rrl_window_size = 2000, rrl_mask_width = 100, species = "CIalpha", z = 0, max_order = 1):
        """
        :rrl_window_size: Units of km/s
        :param rrl_mask_width: Units of km/s
        :param species: i.e. "CIalpha"
        :param z: redshift
        """
        rrl_window_size = rrl_window_size * u.km
        rrl_mask_width = rrl_mask_width * u.km
        rrl_spec_list = []
        stat_list = []

        for spec in spec_list:
                freq = spec[:,1] * u.Hz
                flx = spec[:,2]
                rrl_window = extract_rrl(freq, flx, rrl_window_size, species, z)
                residuals, min_order, min_rchi = minimize_order_baseline(rrl_window, rrl_mask_width, max_order)
                rms = np.ma.std(center_mask(residuals, rrl_mask_width))

                rrl_spec_list.append([rrl_window[0], rrl_window[1], residuals])
                stat_list.append([min_order, min_rchi, rms])

        return rrl_spec_list, stat_list