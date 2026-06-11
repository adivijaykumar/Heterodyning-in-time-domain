#!/usr/bin/env python
# coding: utf-8
import bilby
import sys
import numpy as np

"""
PSD padding:
------------
This script is used to identify the points where the PSD in infinite and pads them with a large value (maximum or a few times the maximum of PSD). The final frequency and PSDs are saved to a text file. 

The resulting file can be used to construct covariance matrices and to generate noise realisations for injection studies.
"""

ifo = bilby.gw.detector.get_empty_interferometer("H1")
ifo.minimum_frequency = 0

duration = 2 # seconds
sampling_frequency = 4096  # Hz

ifo.set_strain_data_from_power_spectral_density(
    sampling_frequency=sampling_frequency,
    duration=duration
)
time_series = ifo.time_array
noise_series = ifo.strain_data.time_domain_strain
psd = ifo.power_spectral_density_array
freq = ifo.frequency_array
idx = np.where(freq<20)[0]
# idx = np.where(psd==np.inf)[0]
psd[idx] = 0
psd[idx] = np.max(psd)

frequencies = freq.copy()
psd_values = psd.copy()
print(psd_values, frequencies)
psd_table = np.column_stack((frequencies, psd_values))
np.savetxt("aLIGO_psd_padded.txt", psd_table, fmt="%.16e")
#np.savetxt("aVirgo_psd_padded.txt", psd_table, fmt="%.16e")
