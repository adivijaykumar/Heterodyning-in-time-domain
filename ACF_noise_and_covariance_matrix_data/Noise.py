#!/usr/bin/env python
# coding: utf-8
import bilby
import sys

"""
Noise Realisation for Injection studies:
----------------------------------------
This script generates synthetic stationary Gaussian noise realisation for specified gravitational wave detectors. User can use any psd files by simply updating the "Psd_files" dictionary with filenames against the corresponding detector.

Important Note: The PSD files mentioned in this script contain power spectrak densities that have been padded with large values below a minimum frequency of 10 Hz.
"""

Psd_files = {"H1":"aLIGO_psd_padded.txt", "L1":"aLIGO_psd_padded.txt","V1":"aVirgo_psd.txt"}

Noise = {}

def noise_realisation_from_psd(length_of_noise_segment, Detectors_list, sampling_frequency = 4096):
    for k in Detectors_list.keys():
        ifo = bilby.gw.detector.get_empty_interferometer(k)
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(psd_file=Psd_files[k])
        ifo.minimum_frequency = 0

        ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=sampling_frequency,
            duration=length_of_noise_segment
        )
        noise_series = ifo.strain_data.time_domain_strain
        Noise[ifo.name] = noise_series[0:length_of_noise_segment*sampling_frequency]
    return Noise
