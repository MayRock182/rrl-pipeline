import numpy as np
from numpy.polynomial.polynomial import Polynomial
from rrl_pipeline.utils import center_mask, rchi
from rrlpy import freq as frequency
from astropy import units as u

def subtrack_baseline(vel, flux, rrl_mask_width = 100 * u.km/u.s, order = 1):
	"""
	Returns baseline-subtracted spectrum
	"""
	print(vel)
	mask = center_mask(vel, rrl_mask_width)
	vel = vel[mask]
	flux = flux[mask]

	pfit = Polynomial.fit(vel, flux, order)
	residuals = flux - pfit

	return residuals

def minimize_order_baseline(vel, flux, rrl_mask_width = 100 * u.km/u.s, max_order = 1):
	min_residuals = []
	min_rchi = 1000
	order = max_order

	for order in range(max_order + 1):
		residuals = subtrack_baseline(vel, flux, rrl_mask_width, order)
		rchi_i = rchi(residuals)

		if rchi_i < min_rchi:
			min_residuals = residuals
			min_order = order
			min_rchi = rchi_i

	return min_residuals, min_order, min_rchi

def extract_rrl(freq, flux, species, z, rrl_window_size = 1000 * u.km/u.s):
	"""
	Finds RRLs found in freq. Returns all RRL spectra found within rrl_window_size.

	:param freq: Frequency vector
	:param flux: Description
	:param species: Description
	:param z: Description
	:param rrl_window_size: Description
	"""
	species_str = "RRL_" + species
	print("First freq:", freq)
	n_array, f_array = frequency.find_lines_sb(np.sort(freq.to("MHz").value), species_str, z=z)
	rrl_windows = []
	#print("n found:", np.size(n_array))
	for n, f in zip(n_array, f_array):
		freq_to_vel = u.doppler_relativistic(freq)
		vel = freq.to(u.km / u.s, equivalencies = freq_to_vel)
		print("Vel:", vel)
		# Find the indices of the range ends.
		# This works bc the center 0 km/s are RRLs
		print("Window size:", rrl_window_size)
		#print("Input low:", abs(vel - rrl_window_size/2))
		#print("Input high:", abs(vel + rrl_window_size/2))
		#print("Low:", np.min(abs(vel - rrl_window_size/2)))
		#print("High:", np.min(abs(vel + rrl_window_size/2)))
		idx_low = np.argmin(abs(vel - rrl_window_size/2))# argmin is used because they are shifting vel
		idx_hgh = np.argmin(abs(vel + rrl_window_size/2))
		idx_low, idx_hgh = np.sort([idx_low, idx_hgh]) # certain that  idx low to high
		#print("idx_low:", idx_low)
		#print("idx_high:", idx_hgh)
		idx_hgh += 1 # Python excludes the upper end of the range.

		rrl_window_frq = freq[idx_low:idx_hgh]
		rrl_window_vel = vel[idx_low:idx_hgh]
		rrl_window_flx = flux[idx_low:idx_hgh]
		print(np.shape(rrl_window_frq))
		print(np.shape(rrl_window_vel))
		print(np.shape(rrl_window_flx))
		rrl_windows.append((rrl_window_frq, rrl_window_vel, rrl_window_flx))
		
	return rrl_windows

def extract_rrls(spec_list, species, z, rrl_window_size = 1000 * u.km/u.s, rrl_mask_width = 100 * u.km/u.s, max_order = 1):
	"""
	Returns a list of rrl spectra found with their baseline subtracted and their corresponding statistics.

	:param spec_list:       Spectrum list with header as [pix,freq (Hz), flx/area (Jy/beam)]. Requires an exluded header row
	:param rrl_window_size: Units of km/s. Window for fitting spectrum
	:param rrl_mask_width:  Units of km/s. Masks center rrl channels for fitting
	:param species:         i.e. "CIalpha"
	:param z:               redshift
	"""
	rrl_spec_list = []
	stat_list = []
	
	for spec in spec_list:
		print(np.shape(spec))
		freq = spec[:,1]
		flx = spec[:,2]
		rrl_window = extract_rrl(freq, flx, species, z, rrl_window_size)
		print(np.shape(rrl_window))
		residuals, min_order, min_rchi = minimize_order_baseline(rrl_window[1], rrl_window[2], rrl_mask_width, max_order)
		rms = np.ma.std(center_mask(residuals, rrl_mask_width))

		rrl_spec_list.append([rrl_window[0], rrl_window[1], residuals])
		stat_list.append([min_order, min_rchi, rms])

	return rrl_spec_list, stat_list