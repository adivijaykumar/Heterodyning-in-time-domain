#!/usr/bin/env python
# coding: utf-8
import bilby
import sys
import numpy as np
from scipy.linalg import toeplitz, inv
import dill
from scipy.linalg import solve_toeplitz
import scipy.signal as sig
from pycbc.detector import Detector

"""
This script computes vectors x and y, required to compute the inverse of Toeplitz covariance matrix using Gohberg-Semencul method (See https://arxiv.org/pdf/2601.11239). These vectors (x and y) are passed into Time-domain Heterodyned likelihood classes, to compute the matrix vector product of form $C^{-1}h$ efficiently. For more information, see https://arxiv.org/pdf/2601.11239.

"""

Detectors_list = {"L1":Detector("L1"),"H1":Detector("H1"), "V1":Detector("V1")}

def x_and_y_for_C_inv(acf):
    e1 = np.zeros(len(acf))
    e1[0] = 1
    x = solve_toeplitz((acf, acf), e1)
    y = x[::-1]
    return x,y

def compute_autocorrelation(d):
    """
    d : 1D array
        Input time-domain signal (strain data or noise).
    """
    n = len(d)
    rho = sig.correlate(d, d)
    rho = np.fft.ifftshift(rho)
    rho = rho[:n] / len(d)
    return rho

def acf_from_psd(psd_data,fs):
    delta_t = 1/fs
    rho = 0.5 * np.real(np.fft.irfft(a=psd_data,)) / (delta_t)
    return rho

x = {}
y = {}

length = 1024
sampling_frequency = 4096  # in Hz
duration = 2 # length of data segment in seconds

for k in Detectors_list.keys():
    ifo = bilby.gw.detector.get_empty_interferometer(k)
    # ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(psd_file="aLIGO_psd.txt")
    ifo.minimum_frequency = 0
    print(ifo)

    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=sampling_frequency,
        duration=length
    )

    time_series = ifo.time_array
    noise_series = ifo.strain_data.time_domain_strain
    psd = ifo.power_spectral_density_array
    freq = ifo.frequency_array
    idx = np.where(freq<20)[0]
    # idx = np.where(psd==np.inf)[0]
    psd[idx] = 0
    psd[idx] = np.max(psd)
    ACF = acf_from_psd(psd, fs=4096)
    x[k], y[k] = x_and_y_for_C_inv(ACF[0:duration*sampling_frequency])

Inv_cov_Noise = {"x":x, "y":y}
with open("x_and_y_" + str(duration) + "_sec.pkl", "wb") as f:
    dill.dump(Inv_cov_Noise, f)
