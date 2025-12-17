import numpy as np
from numpy.polynomial.polynomial import Polynomial
from rrl_pipeline.utils import center_mask, rchi
from rrlpy import freq as frequency
from astropy import units as u
from astropy import table
from astropy.modeling import models, fitting

def subtract_baseline(window, rrl_mask_width = 100 * u.km/u.s, order = 1):
	"""
	Returns baseline-subtracted spectrum. Can apply masking for RRL channels
	"""
	vel = window["Velocity"].copy()
	flux = window["Flux"].copy()
	#print("Final velocity:", len(vel))
	mask = center_mask(vel, rrl_mask_width)
	#print("Mask:", len(mask))
	#print("Unmask velocity:", len(vel.unmasked))
	#print("Mask flux:", len(flux))
	vel.mask = ~mask
	flux.mask = ~mask

	poly_init = models.Polynomial1D(degree=order)
	fitter = fitting.LinearLSQFitter(calc_uncertainties=True)
	pfit = fitter(poly_init, vel[mask], flux[mask])
	print("Vel Masked:", len(pfit(vel[mask])))
	residuals = flux - pfit(vel)
	print("Residual Type:", type(residuals))
	return residuals

def minimize_order_baseline(rrl_window, rrl_mask_width = 100 * u.km/u.s, max_order = 1):
	print("Processing RRL#", rrl_window.meta["RRL Number"])
	min_residuals = []
	min_rchi = 1000
	min_order = max_order

	for order in range(max_order+1):
		residuals = subtract_baseline(rrl_window, rrl_mask_width, order)
		rchi_i = rchi(residuals, order)
		print("Temp Residual", residuals)
		if rchi_i < min_rchi:
			min_residuals = residuals
			min_order = order
			min_rchi = rchi_i
	
	if min_residuals == []:
		min_residuals = residuals.copy()

	print("Rchi minimized to:", min_rchi)
	print("Minimized Order:", min_order)
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
	#print("First freq:", freq)
	# TODO Maybe check if frequency.find_lines_sb can work w/ astropy tables. 
	# 	   Add .value in line where error occurs?
	n_array, f_array = frequency.find_lines_sb(np.sort(freq.to("MHz")).value, species_str, z=z)
	f_array *= u.MHz
	rrl_windows = []
	#print("f found:", f_array)
	for n, f in zip(n_array, f_array):
		print("Found RRL #",int(n))
		freq_to_vel = u.doppler_relativistic(f)
		vel = freq.to(u.km / u.s, equivalencies = freq_to_vel)
		#print("Vel:", vel)
		# Find the indices of the range ends.
		# This works bc the center 0 km/s are RRLs
		#print("Window size:", rrl_window_size)
		#print("Input low:", abs(vel - rrl_window_size/2))
		#print("Input high:", abs(vel + rrl_window_size/2))
		#print("Low:", np.min(abs(vel - rrl_window_size/2)))
		#print("High:", np.min(abs(vel + rrl_window_size/2)))
		idx_low = abs(vel - rrl_window_size/2).argmin()# argmin is used because they are shifting vel
		idx_hgh = abs(vel + rrl_window_size/2).argmin()
		idx_low, idx_hgh = np.sort([idx_low, idx_hgh]) # certain that  idx low to high
		#print("idx_low:", idx_low)
		#print("idx_high:", idx_hgh)
		idx_hgh += 1 # Python excludes the upper end of the range.

		rrl_window_frq = freq[idx_low:idx_hgh]
		rrl_window_vel = vel[idx_low:idx_hgh]
		rrl_window_flx = flux[idx_low:idx_hgh]
		#print(rrl_window_frq)
		#print(rrl_window_vel)
		#print(rrl_window_flx)
		rrl_window = table.QTable([rrl_window_frq, rrl_window_vel, rrl_window_flx], 
								  masked=True,
								  names=["Frequency", "Velocity", "Flux"],
								  units=[u.Hz, u.km/u.s, u.Jy/u.beam],
								  meta={"RRL Number":int(n), "Frequency":f})
		
		rrl_windows.append(rrl_window)
		
	return rrl_windows

def extract_rrls(spec_list, species, z, rrl_window_size = 1000 * u.km/u.s, rrl_mask_width = 100 * u.km/u.s, max_order = 1):
	"""
	Returns a list of rrl spectra found with their baseline subtracted and their corresponding statistics.

	:param spec_list:       List of spectra, each as an Astropy QTable with columns 'frequency' (Hz) and 'flux' (Jy/beam).
	:param rrl_window_size: Units of km/s. Window for fitting spectrum
	:param rrl_mask_width:  Units of km/s. Masks center rrl channels for fitting
	:param species:         i.e. "CIalpha"
	:param z:               redshift
	"""
	print("Number of Windows:", len(spec_list))
	rrl_spec_list = []
	stat_list = []
	
	for i, spec in enumerate(spec_list):
		print("Processing Spec #",i+1)
		#print(np.shape(spec))
		rrl_windows = extract_rrl(spec["frequency"], 
						   		 spec["flux"], 
								 species, 
								 z, 
								 rrl_window_size)
		
		for window in rrl_windows:
			print(np.shape(window))
			residuals, min_order, min_rchi = minimize_order_baseline(window,
																	 rrl_mask_width, 
																	 max_order)
			mask = center_mask(window["Velocity"], rrl_mask_width)
			print("Mask:", mask)
			print("Residuals:",residuals)
			print("Flux:",window["Flux"])
			# Mask RRL channels for determining RMS and ~mask for the signal
			residuals_rrl_mask = residuals[mask]
			rms = residuals_rrl_mask.std()
			signal = abs(residuals[~mask]).max()
			
			window["Flux Residuals"] = residuals
			window.meta.update({"Order":min_order, 
					   			"Red. Chi Square":min_rchi, 
								"RMS":rms, 
								"Signal":signal})

			rrl_spec_list.append(window)

	return rrl_spec_list