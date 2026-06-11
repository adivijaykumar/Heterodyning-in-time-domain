#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
from pycbc.detector import Detector
import bilby
import json
import dill
# from pycbc.waveform import get_td_waveform
# import bilby
# from gwpy.timeseries import TimeSeries
from Relative_22_row_new_2 import *
# from scipy.linalg import toeplitz, inv, circulant
# from scipy.fft import fft, ifft
# import h5py
# from scipy.linalg import toeplitz, inv
# import scipy.signal as sig
# from pycbc.detector import Detector
# import dill
# from scipy.signal import welch
# from gwpy.timeseries import TimeSeries
# from gwosc.datasets import event_gps
# from scipy.signal.windows import tukey
# from pycbc.filter import highpass, lowpass
import ringdown as rd
import matplotlib.pyplot as plt
from bilby.gw.conversion import component_masses_to_chirp_mass, chirp_mass_and_mass_ratio_to_component_masses
logger = bilby.core.utils.logger

outdir = "outdir_NOV_19_170817_0.05_0.05_20_Hz_2047_Hz_ringdown_conditioning"
label = "GW170817"

trigger_time = 1187008882.4
duration = 128
post_trigger_duration = 0.05
minimum_frequency = 20
maximum_frequency = 2048
Time_array = np.arange(-duration+post_trigger_duration,post_trigger_duration,1/4096)
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

Detectors_list = {"V1":Detector("V1"), "H1":Detector("H1"),"L1":Detector("L1"),}

fs = 4096
with open(
    "Data_ringdown_conditioned_GW170817_128_sec.pkl",
    "rb",
) as f:
    Data_list = dill.load(f)

def acf_from_psd(psd_data,fs,N):
    delta_t = 1/fs
    rho = 0.5 * np.real(np.fft.irfft(a=psd_data,)) / (delta_t)
    return rho

def inverse_circulant_first_row(c):
    eigenvalues = fft(c)
    if np.any(eigenvalues == 0):
        raise ValueError("The matrix is singular and does not have an inverse.")
    inv_eigenvalues = 1 / eigenvalues
    c_inv_first_row = ifft(inv_eigenvalues).real
    return c_inv_first_row

C_inv_row = {}
for k in Detectors_list.keys():
    ifo = bilby.gw.detector.get_empty_interferometer(k)
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(psd_file=str(k)+"_GW170817_ligo_psd.txt")
    ifo.minimum_frequency = 0
    print(ifo)

    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=4096,
        duration=128
    )
    psd = ifo.power_spectral_density_array
    freq = ifo.frequency_array
    idx = np.where(psd==np.inf)[0]
    psd[idx] = 1e-35
    ACF = acf_from_psd(psd,4096,duration*4096)
    C_inv_row[k] = inverse_circulant_first_row(ACF)

print("PSD : ",psd[0], psd[-1])

fiducial_parameters = {
    "chi_1": 0.018,
    "chi_2": 0.016,
    "chirp_mass": 1.1975,
    "dec": -0.408084,
    "geocent_time": trigger_time,
    "luminosity_distance": 40,
    "mass_ratio": 0.92,
    "phase": 3.0136566567608765,
    "psi": 0.81,
    "ra": 3.44616,
    "theta_jn": 2.69,
}

print(fiducial_parameters)
priors = bilby.gw.prior.CBCPriorDict(
    dict(
        chirp_mass=bilby.core.prior.Uniform(
            minimum=1.18,
            maximum=1.21,
            name="chirp_mass",
            latex_label="$\\mathcal{M}$",
        ),
        mass_ratio=bilby.core.prior.Uniform(
            minimum=1 / 6,
            maximum=1.0,
            name="mass_ratio",
            latex_label="$q$",
        ),
        chi_1=bilby.core.prior.Uniform(
            minimum=-0.05,
            maximum=0.05,
            name="chi_1",
            latex_label="$\\chi_1$",
        ),
        chi_2=bilby.core.prior.Uniform(
            minimum=-0.05,
            maximum=0.05,
            name="chi_2",
            latex_label="$\\chi_2$",
        ),
        luminosity_distance=bilby.gw.prior.UniformSourceFrame(
            name="luminosity_distance", minimum=10, maximum=300, unit="Mpc"
        ),
        theta_jn=bilby.core.prior.Sine(name="theta_jn"),
        ra=fiducial_parameters["ra"],
        dec=fiducial_parameters["dec"],
        psi=bilby.core.prior.Uniform(
            minimum=0, maximum=np.pi, name="psi", boundary="periodic"
        ),
        phase=bilby.core.prior.Uniform(
            minimum=0, maximum=2 * np.pi, name="phase", boundary="periodic"
        ),
        geocent_time=bilby.core.prior.Uniform(
            trigger_time - 0.04,
            trigger_time + 0.04,
            name="geocent_time",
            latex_label="$t_{c}$",
            boundary=None,
        ),
    )
)


epsilon1_H1, epsilon2_H1 = 0.05,0.05 #float(sys.argv[3]),float(sys.argv[4])
epsilon1_L1, epsilon2_L1 = 0.05,0.05 #float(sys.argv[5]),float(sys.argv[6])
epsilon1_V1, epsilon2_V1 = 0.05,0.05 #float(sys.argv[7]),float(sys.argv[8])

spacing1, spacing2, spacing3,spacing_times1,spacing_times2 = 1, 100, 100,0.05, 0.1 #int(sys.argv[9]),int(sys.argv[10]),int(sys.argv[11]), float(sys.argv[12]),float(sys.argv[13])

spacing = [spacing1, spacing2, spacing3]
epsilon = {"H1":np.array([epsilon1_H1,epsilon2_H1]),"L1":np.array([epsilon1_L1,epsilon2_L1]),"V1":np.array([epsilon1_V1,epsilon2_V1])}
spacing_times = [spacing_times1, spacing_times2]
print("epsilon,spacing,spacing_times",epsilon, spacing,spacing_times)

# Defining the time array:
print("t[0]:", Time_array[0])
print("t[-1]:",Time_array[-1])


Likelihood_obj = RelativeBinningTimeDomain_H1detectorframe_for_longer_signals(
    time = Time_array,
    Detectors_list = Detectors_list,
    fiducial_parameters = fiducial_parameters,
    injection_parameters = None,
    Data_list = Data_list,
    C_inv_row = C_inv_row,
    Noise = {},
    trigger_time = trigger_time,
    fmin = minimum_frequency,
    fref = 20,
    epsilon_array_per_detector=epsilon,
    spacing=spacing,
    spacing_times=spacing_times,
)

Likelihood_obj.parameters.update(fiducial_parameters)
print("Fiducial likelihood:",Likelihood_obj.log_likelihood_ratio())

result = bilby.run_sampler(
    likelihood=Likelihood_obj,
    priors=priors,
    nlive=500,
    label=label,
    outdir=outdir,
    sampler="dynesty",
    sample="acceptance-walk",
    naccept=60,
    npool=32,
)

