import numpy as np
from numpy.polynomial.polynomial import Polynomial
from rrl_pipeline.utils import center_mask, rchi2, quality_factor
from rrlpy import freq as frequency
from astropy import units as u
from astropy import table
from astropy.modeling import models, fitting

# TODO Make an rrl_window class. rrl_mask_width and rrl_window_size as specs would be useful

def subtract_baseline(window, rrl_mask_width = 100 * u.km/u.s, order = 1):
	"""
	Returns baseline-subtracted spectrum. Can apply masking for RRL channels
	"""
	vel = window["Velocity"].copy()
	flux = window["Flux"].copy()
	#print("Final velocity:", len(vel))
	mask_rrl = center_mask(vel, rrl_mask_width)
	mask_nonzero = flux != 0
	combined_mask = mask_rrl & mask_nonzero
	#print("Mask:", combined_mask)
	#print("Velocity:", vel.mask)
	vel.mask = ~combined_mask
	flux.mask = ~combined_mask
	#print("Mask:", combined_mask)
	#print("Velocity:", vel)
	#print("After Flux:", flux)
	
	poly_init = models.Polynomial1D(degree=order)
	fitter = fitting.LinearLSQFitter(calc_uncertainties=True)
	pfit = fitter(poly_init, vel, flux)

	model = pfit(vel)
	model[~mask_nonzero] = 0
	#print("Model", model.unmasked) # TODO FIX DISCREPENCY!!!
	model.mask = combined_mask
	residuals = flux - model # residual type is MaskedQuantity
	#print("Vel Masked:", len(pfit(vel[mask])))
	#print("Residuals:",residuals.unmasked)
	
	return residuals

def minimize_order_baseline(rrl_window, rrl_mask_width = 100 * u.km/u.s, max_order = 1):
	print("Processing RRL n =", rrl_window.meta["RRL Number"])
	residuals0 = subtract_baseline(rrl_window, rrl_mask_width, 0)
	min_residuals = residuals0
	min_rchi = rchi2(residuals0, 0)
	min_order = 0

	for order in range(1, max_order+1):
		residuals = subtract_baseline(rrl_window, rrl_mask_width, order)
		rchi_i = rchi2(residuals, order)
		
		if abs(1 - min_rchi) > abs(1 - rchi_i):
			min_residuals = residuals
			min_order = order
			min_rchi = rchi_i
	
	if min_residuals is None:
		min_residuals = residuals.copy()

	#print("Rchi minimized to:", min_rchi)
	#print("Minimized Order:", min_order)
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
	# TODO Add 0 padding for wedge cases where spectrum is at the edge of the subband
	n_array, f_array = frequency.find_lines_sb(np.sort(freq.to("MHz")).value, species_str, z=z)
	f_array *= u.MHz
	rrl_windows = []
	#print("f found:", f_array)
	for n, f in zip(n_array, f_array):
		print("Found RRL n =",int(n))
		# Use center frequency, f, for doppler shift from freq to vel
		freq_to_vel = u.doppler_relativistic(f)
		vel = freq.to(u.km / u.s, equivalencies = freq_to_vel)
		
		#print("Vel:", vel)
		
		# Finding the indices of the range ends.
		half_window = rrl_window_size/2
		
		# TODO turn into function to handle edge cases
		# Handle edge case when vel does not encompass rrl_window_size centered at 0 km/s
		# If positive, padding is needed
		missing_high = -(vel.max() - half_window) 
		missing_low = vel.min() + half_window
		
		if missing_high > 0:
			print("Missing Upper", missing_high)
			df = np.diff(freq)[0] # freq is evenly spaced
			freq_target_max = half_window.to(u.MHz, equivalencies=freq_to_vel)
			#print("Freq:", freq.to(u.MHz))
			#print("Center", f)
			#print("Data f:", freq.min().to(u.MHz))
			#print("Pad to freq",freq_target_max)
			#print("Difference:", (freq.min()-freq_target_max).to(u.MHz))
			#print("f Chan Size:",df.to(u.MHz))
			n_pad_high = int(np.ceil((freq.min() - freq_target_max)/df))
			#print("Add padding", n_pad_high)
			freq_high_pad = freq.min() - np.arange(n_pad_high, 0, -1) * df
			freq = np.concatenate([freq_high_pad, freq])
			#print("Padded Freq:", freq.to(u.MHz))
			#print(flux)
			flux = np.concatenate([np.zeros(n_pad_high), flux]) # TODO check
			#print(flux)
		if missing_low > 0:
			print("Missing Lower", missing_low)
			#print("Hello???")
			df = np.diff(freq)[0]
			freq_target_min = (-half_window).to(u.Hz, equivalencies=freq_to_vel)
			n_pad_low = int(np.ceil((freq_target_min - freq.max())/df)) # TODO need to check if np.ceil allows it to reach +-500
			#print("Add padding", n_pad_low)
			freq_low_pad = freq.max() + np.arange(1, n_pad_low+1) * df
			#print("Padded Freq:", freq.to(u.MHz))
			freq = np.concatenate([freq, freq_low_pad])
			flux = np.concatenate([flux, np.zeros(n_pad_low)])

		#print("Should be the same", freq.to(u.MHz)[:10])
		vel = freq.to(u.km/u.s, equivalencies=freq_to_vel)
		#print("Vel After Potential Padding:", vel)

		idx_low = np.where(vel < -rrl_window_size/2)[0].min()
		idx_high = np.where(vel > rrl_window_size/2)[0].max()

		idx_low, idx_high = np.sort([idx_low, idx_high]) # certain that  idx low to high
		idx_high += 1 # Python excludes the upper end of the range.

		rrl_window_frq = freq[idx_low:idx_high]
		rrl_window_vel = vel[idx_low:idx_high]
		rrl_window_flx = flux[idx_low:idx_high]
		#print("RRL Window Velocity:",rrl_window_vel)
		rrl_window = table.QTable([rrl_window_frq, rrl_window_vel, rrl_window_flx], 
								  masked=True,
								  names=["Frequency", "Velocity", "Flux"],
								  units=[u.Hz, u.km/u.s, u.Jy/u.beam],
								  meta={"RRL Number":int(n), "Frequency":f})
		
		rrl_windows.append(rrl_window)
		
	return rrl_windows

def extract_rrls(spec_list, species, z, rrl_window_size=1000 * u.km/u.s, rrl_mask_width=100 * u.km/u.s, max_order = 1):
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
	
	for i, spec in enumerate(spec_list):
		print("Processing Subband #",i+1)
		
		rrl_windows = extract_rrl(spec["frequency"], 
						   		 spec["flux"], 
								 species, 
								 z, 
								 rrl_window_size)
		
		for window in rrl_windows:
			#print(window["Flux"])
			residuals, min_order, min_rchi = minimize_order_baseline(window,
																	 rrl_mask_width, 
																	 max_order)
			# Mask RRL channels for determining RMS and ~mask for the signal
			mask = center_mask(window["Velocity"], rrl_mask_width)
			rms = residuals.std() # already has mask
			#print(print("Residuals Signal:",residuals[~mask].unmasked))
			signal = abs(residuals[~mask].unmasked).max()
			quality = quality_factor(residuals)
			
			window["Flux Residuals"] = residuals
			window.meta.update({"Order":min_order, 
					   			"Red. Chi Square":min_rchi, 
								"RMS":rms, 
								"Signal":signal,
								"Quality":quality
								})

			rrl_spec_list.append(window)

		print("Finished Subband #", i+1)
		print("\n")

	return rrl_spec_list