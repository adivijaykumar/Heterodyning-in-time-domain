#!/usr/bin/env python
# coding: utf-8
import sys
import numpy as np
from pycbc.detector import Detector
import bilby
import json
import dill
from pycbc.waveform import get_td_waveform
import bilby
from gwpy.timeseries import TimeSeries
from Relative_22_row import *
from scipy.linalg import toeplitz, inv, circulant
from scipy.fft import fft, ifft
import h5py
import matplotlib.pyplot as plt
from bilby.gw.conversion import component_masses_to_chirp_mass, chirp_mass_and_mass_ratio_to_component_masses
logger = bilby.core.utils.logger

outdir = "outdir"
label = "GW190425"

trigger_time = 1240215503.0
detectors = ["L1","V1"]
maximum_frequency = 512
minimum_frequency = 20
roll_off = 0.4 
duration = 128
post_trigger_duration = 1/4096
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

fiducial_parameters = {
    "chirp_mass": 1.48658,
    "mass_ratio": 0.8955,
    "chi_1": 0.018,
    "chi_2": 0.016,
    "luminosity_distance": 206.751,
    "theta_jn": 0.185,
    "ra": 4.2,
    "dec": 0.438,
    "phase": 3.0136566567608765,
    "psi": 0.281,
    "geocent_time": 1240215503.039 
}

L1_ifo = bilby.gw.detector.get_empty_interferometer("L1")
to_L1 = L1_ifo.time_delay_from_geocenter(
    ra = fiducial_parameters["ra"],
    dec = fiducial_parameters["dec"],
    time = fiducial_parameters["geocent_time"]
    )
fiducial_parameters["L1_time"] = fiducial_parameters["geocent_time"] + to_L1
fiducial_parameters.pop("geocent_time")

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

psd_duration = 1024
psd_start_time = start_time - psd_duration
psd_end_time = start_time

Data_list = {}
C_inv_row = {}

for det in detectors:
    fname = f"{det}_GWOSC.hdf5"
    with h5py.File(fname, "r") as f:
        data = f["strain/Strain"][:]
        # print(data)
        dt = f["strain/Strain"].attrs["Xspacing"]
        t_start = f["meta/GPSstart"][()] 
        full_duration = f["meta/Duration"][()]
        t_end = t_start + full_duration
        t = np.arange(t_start, t_end, dt)
        idx = np.where((t >= start_time) & (t < end_time))[0]
        # print(data)
        # data = TimeSeries(data, t0 = start_time, dt = dt)
        # data = data.highpass(10).lowpass(1300)
        Data_list[det] = data[idx]
        
        idx2 = np.where((t >= psd_start_time) & (t < psd_end_time))[0]
        psd_data = data[idx2]
        psd_alpha = 2 * roll_off / duration
        noise = TimeSeries(psd_data, t0 = psd_start_time, dt = dt)
        # noise = noise.highpass(10).lowpass(1350)
        psd = noise.psd(
            fftlength=duration, overlap=4, window=("tukey", psd_alpha), method="median"
        )
        idx = np.where(psd.frequencies.value <= 10)[0]
        psd[idx] = 10*np.max(psd)
        idx2 = np.where(psd.frequencies.value >= 1350)[0]
        psd[idx2] = 1e-6*np.max(psd)
        plt.loglog(psd, label = det)
        ACF = acf_from_psd(psd,4096,1)
        C_inv_row[det] = inverse_circulant_first_row(ACF)

# with open(
#     "C_inv_row_GW190425.pkl",
#     "rb",
# ) as f:
#     Inv_cov_Noise = dill.load(f)
# C_inv_row = Inv_cov_Noise
# print("C_inv_row : ", C_inv_row)

logger.info("Saving data plots to {}".format(outdir))
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)

epsilon1_H1, epsilon2_H1 = 0.05,0.05 #float(sys.argv[3]),float(sys.argv[4])
epsilon1_L1, epsilon2_L1 = 0.05,0.05 #float(sys.argv[5]),float(sys.argv[6])
epsilon1_V1, epsilon2_V1 = 0.05,0.05 #float(sys.argv[7]),float(sys.argv[8])

spacing1, spacing2, spacing3,spacing_times1,spacing_times2 = 1, 10, 100,0.02, 0.05 #int(sys.argv[9]),int(sys.argv[10]),int(sys.argv[11]), float(sys.argv[12]),float(sys.argv[13])

spacing = [spacing1, spacing2, spacing3]
epsilon = {"H1":np.array([epsilon1_H1,epsilon2_H1]),"L1":np.array([epsilon1_L1,epsilon2_L1]),"V1":np.array([epsilon1_V1,epsilon2_V1])}
spacing_times = [spacing_times1, spacing_times2]
print("epsilon,spacing,spacing_times",epsilon, spacing,spacing_times)

# Defining the time array:

Time_array = np.arange(start_time-trigger_time, end_time-trigger_time, 1 / 4096.0, dtype=np.float64)
print("t[0]:", Time_array[0])
print("t[-1]:",Time_array[-1])

Detectors_list = {"L1":Detector("L1"),"V1":Detector("V1")}

Likelihood_obj = RelativeBinningTimeDomain_H1detectorframe_for_longer_signals(
    time = Time_array,
    Detectors_list = Detectors_list,
    fiducial_parameters = fiducial_parameters,
    injection_parameters = None,
    Data_list = Data_list,
    C_inv_row = C_inv_row,
    Noise = {},
    trigger_time = trigger_time,
    detector_frame = "L1",
    epsilon_array_per_detector=epsilon,
    spacing=spacing,
    spacing_times=spacing_times,
)
SNR, Network_SNR = Likelihood_obj.compute_SNR_TD_and_waveform_data()[0:2]

priors = bilby.gw.prior.CBCPriorDict(
    dict(
        chirp_mass=bilby.core.prior.Uniform(
            minimum=1.485,
            maximum=1.49,
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
        ra=bilby.core.prior.Uniform(
            minimum=0, maximum=2 * np.pi, name="ra", boundary="periodic"
        ),
        dec=bilby.core.prior.Cosine(name="dec"),
        psi=bilby.core.prior.Uniform(
            minimum=0, maximum=np.pi, name="psi", boundary="periodic"
        ),
        phase=bilby.core.prior.Uniform(
            minimum=0, maximum=2 * np.pi, name="phase", boundary="periodic"
        ),
        L1_time=bilby.core.prior.Uniform(
            trigger_time - 0.005,
            trigger_time + 0.005,
            name="L1_time",
            latex_label="$t_{L1}$",
            unit=None,
            boundary=None,
        ),
    )
)

Maximum_likelihood_parameters_obj = Set_Fiducial_parameters_for_longer_signals(
    time = Time_array,
    Detectors_list = Detectors_list,
    fiducial_parameters = fiducial_parameters,
    injection_parameters = None,
    Data_list = Data_list,
    C_inv_row = C_inv_row,
    Noise = {},
    trigger_time = trigger_time,
    detector_frame = "L1",
    epsilon_array_per_detector=epsilon,
    spacing=spacing,
    spacing_times=spacing_times,
    priors = priors,
)

new_fiducial_parameters = (
    Maximum_likelihood_parameters_obj.optimize_fiducial_parameters()
)
fiducial_parameters = new_fiducial_parameters.copy()

del Maximum_likelihood_parameters_obj
del new_fiducial_parameters
del Likelihood_obj

Likelihood_obj = RelativeBinningTimeDomain_H1detectorframe_for_longer_signals(
    time = Time_array,
    Detectors_list = Detectors_list,
    fiducial_parameters = fiducial_parameters,
    injection_parameters = None,
    Data_list = Data_list,
    C_inv_row = C_inv_row,
    Noise = {},
    trigger_time = trigger_time,
    detector_frame = "L1",
    epsilon_array_per_detector=epsilon,
    spacing=spacing,
    spacing_times=spacing_times,
)

print("Fiducial_parameters : ", Likelihood_obj.fiducial_parameters)
Likelihood_obj.parameters.update(fiducial_parameters)
print("Fiducial Likelihood : ", Likelihood_obj.log_likelihood_ratio())
# print("Injection_parameters : ", Likelihood_obj.injection_parameters)
print("Number of bins : ",Likelihood_obj.number_of_bins)
print("Fiducial waveform : ", Likelihood_obj.h22_fiducial)
print("Gamma : ", Likelihood_obj.gamma)

mc_min = fiducial_parameters["chirp_mass"] - 0.005/2
mc_max = fiducial_parameters["chirp_mass"] + 0.005/2

priors["chirp_mass"] = bilby.core.prior.Uniform(
    minimum=mc_min,
    maximum=mc_max,
    name="chirp_mass",
    latex_label="$\\mathcal{M}$",
)

result = bilby.run_sampler(
    likelihood=Likelihood_obj,
    priors=priors,
    nlive=500,
    label=label,
    outdir=outdir,
    sampler="dynesty",
    sample="acceptance-walk",
    naccept=60,
    npool=16,
)

