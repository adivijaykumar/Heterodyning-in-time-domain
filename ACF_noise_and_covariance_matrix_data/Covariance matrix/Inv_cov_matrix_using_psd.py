#!/usr/bin/env python
# coding: utf-8
import bilby
import sys
import numpy as np
from scipy.linalg import toeplitz, inv
import dill

"""
Noise and Inverse covariance matrix:
------------------------------------
This script generates gaussian noise and computes the inverse of covariance matrix for specified gravitational wave detectors.

"""

def acf_from_psd(psd_data,fs):
    delta_t = 1/fs
    rho = 0.5 * np.real(np.fft.irfft(a=psd_data,)) / (delta_t)
    return rho

Detectors_list = [bilby.gw.detector.get_empty_interferometer("L1"),bilby.gw.detector.get_empty_interferometer("H1"),bilby.gw.detector.get_empty_interferometer("V1")]

C_inv = {}
Noise = {}

length_of_noise_segment = 1024 # seconds
sampling_frequency = 4096  # Hz

duration = 2 # seconds of data

for ifo in Detectors_list:
    ifo.minimum_frequency = 0
    print(ifo)

    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=sampling_frequency,
        duration=length_of_noise_segment
    )
    
    time_series = ifo.time_array
    noise_series = ifo.strain_data.time_domain_strain
    freq = ifo.frequency_array
    psd = ifo.power_spectral_density_array
    idx = np.where(psd==np.inf)[0]
    psd_2 = psd.copy()
    psd_2[idx] = 0
    psd_2[idx] = np.max(psd_2)
    ACF = acf_from_psd(psd_2,4096)

    C_inv[ifo.name] = inv(toeplitz(ACF[0:duration*4096]))
    Noise[ifo.name] = noise_series[0:duration*4096]

Inv_cov_Noise = {"C_inv": C_inv, "Noise": Noise}
with open("C_inv_" + str(duration) + "_sec.pkl", "wb") as f:
    dill.dump(Inv_cov_Noise, f)
