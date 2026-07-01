#!/usr/bin/env python
# coding: utf-8
<<<<<<< HEAD
"""
Backward-compatibility shim — imports unified class from exact.py.
"""
from exact import (  # noqa: F401
    ExactLikelihoodTimeDomain,
    ExactLikelihoodTimeDomainGeocentTimeFrame,
    ExactLikelihoodTimeDomainH1detectorframe,
)
=======
import sys
import bilby
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
import numpy as np
import lalsimulation as ls
import lal
from scipy.linalg import matmul_toeplitz
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ACF_noise_and_covariance_matrix_data'))
from covariance_solver import ToeplitzSolver

print(ls.__file__)


class ExactLikelihoodTimeDomainGeocentTimeFrame(bilby.Likelihood):
    """
    Parameters
    ----------
    time : 1D array
        An array of time points at which waveform/data is generated.
    Detectors_list : list
        List of detector names (pycbc.detector.Detector)
    injection_parameters : dict
        Dictionary of injection parameters.
    Data_list : list
        List containing the observed data from each detector.
    x : 
    y : 
    Noise : array-like
        Noise properties of the detectors.
    modes_to_activate : list of tuples, optional (default=[(2, 2)])
        List of waveform modes to include in the analysis.
    priors : dict, optional
        Prior distributions for the model parameters.
    """

    def __init__(
        self,
        time,
        Detectors_list,
        injection_parameters,
        Data_list,
        x,
        y,
        Noise,
        fmin=10,
        fref=10,
        modes_to_activate=[(2, 2)],
        solver=None,
    ):
        super().__init__(parameters={})
        self.time = time
        self.Detectors_list = Detectors_list
        self.injection_parameters = injection_parameters.copy()
        self.x = x
        self.y = y
        self.Noise = Noise
        # Build solver dict: use provided solvers or fall back to ToeplitzSolver(x[k], y[k])
        if solver is not None:
            self.solver = solver
        else:
            self.solver = {k: ToeplitzSolver(x[k], y[k]) for k in (Detectors_list or {})}
        self._Data_list = Data_list if Data_list else {}
        self.modes_to_activate = modes_to_activate
        self.noise_log_likelihood_value = None
        self.fmin = fmin
        self.fref = fref

        self.create_lalparams_with_modes()
        self.compute_SNR_TD_and_waveform_data()

    def Inner_product_C_inv_vector(self, x, y, v):
        """Legacy interface: preserved for backward compatibility."""
        return ToeplitzSolver(x, y).apply_C_inv(v)

    @property
    def injection_parameters(self):
        return self._injection_parameters

    @injection_parameters.setter
    def injection_parameters(self, new_params):
        self._injection_parameters = new_params.copy()
        self._Data_list = {}  # Set Data_list = 0 if injection params are changed.

    @property
    def Data_list(self):
        return self._Data_list

    def create_lalparams_with_modes(self):
        lalParams = lal.CreateDict()
        ModeArray = ls.SimInspiralCreateModeArray()
        for l, m in self.modes_to_activate:
            print(l, m)
            ls.SimInspiralModeArrayActivateMode(ModeArray, l, m)
            ls.SimInspiralModeArrayActivateMode(ModeArray, l, -m)
        ls.SimInspiralWaveformParamsInsertModeArray(lalParams, ModeArray)
        self.lalParams = lalParams

    def weighted_inner_product_circulant(self, x, y, h1, h2):
        """
        Compute (h1, C^-1 h2) using FFT
        """
        C_inv_h2 = self.Inner_product_C_inv_vector(x, y, h2)
        return np.dot(h1, C_inv_h2)

    def compute_SNR_TD_and_waveform_data(self):
        """
        Computes SNR
        """
        data_times_C_inv = {}
        data_times_C_inv_times_data = {}
        Antenna_pattern_dict = {}
        h_injection = {}
        Time_array_per_detector = {}
        t_delay_injection_dict = {}
        for k in self.Detectors_list.keys():
            t_delay = self.Detectors_list[k].time_delay_from_earth_center(
                right_ascension=self.injection_parameters["ra"],
                declination=self.injection_parameters["dec"],
                t_gps=self.injection_parameters["geocent_time"],
            )
            Time_array_per_detector[k] = self.time + t_delay
            t_delay_injection_dict[k] = t_delay
            h_injection[k] = self.waveform(
                self.injection_parameters, Time_array_per_detector[k] - t_delay
            )
            F_plus, F_cross = self.Detectors_list[k].antenna_pattern(
                right_ascension=self.injection_parameters["ra"],
                declination=self.injection_parameters["dec"],
                polarization=self.injection_parameters["psi"],
                t_gps=self.injection_parameters["geocent_time"],
            )

            Antenna_pattern_dict[k] = {"F_plus": F_plus, "F_cross": F_cross}

            if (
                k not in self.Data_list
                or self.Data_list[k] is None
                or len(self.Data_list[k]) == 0
            ):
                data = (
                    F_plus * np.real(h_injection[k])
                    - F_cross * np.imag(h_injection[k])
                    + self.Noise[k]
                )
                self.Data_list[k] = data
            data_times_C_inv[k] = self.solver[k].apply_C_inv(self.Data_list[k])
            data_times_C_inv_times_data[k] = np.matmul(
                data_times_C_inv[k], self.Data_list[k]
            )
        self.Time_array_per_detector = Time_array_per_detector
        self.t_delay_injection_dict = t_delay_injection_dict
        self.data_times_C_inv = data_times_C_inv
        self.data_times_C_inv_times_data = data_times_C_inv_times_data
        self.h_injection = h_injection
        SNR_dict = {}
        SNR_arr = []
        for k in self.Detectors_list.keys():
            signal = Antenna_pattern_dict[k]["F_plus"] * np.real(
                self.h_injection[k]
            ) + Antenna_pattern_dict[k]["F_cross"] * (-np.imag(self.h_injection[k]))
            hh = np.dot(signal, self.solver[k].apply_C_inv(signal))
            dh = np.dot(self.Data_list[k], self.solver[k].apply_C_inv(signal))
            SNR = np.abs(dh) / np.sqrt(np.abs(hh))
            SNR_arr.append(SNR)
            SNR_dict[k] = SNR
        return (
            SNR_dict,
            np.sqrt(np.sum(np.array(SNR_arr) ** 2)),
            self.h_injection,
            self.Data_list,
            self.data_times_C_inv,
            self.data_times_C_inv_times_data,
        )

    def waveform(self, parameters, Time_array, fmin=None, fref=None, dT=1 / 4096.0):
        """
        Computes time-domain waveform (approximant - IMRPhenomTHM).

        Parameters
        ----------
        parameters : dict
            A dictionary containing the parameters of binary system.
        time : 1D array
            A numpy array of time points at which the hlm mode is to be computed.

        Returns
        -------
        h : 1D array
            time-domain waveform
        """
        if fmin is None:
            fmin = self.fmin
        if fref is None:
            fref = self.fref
        chirp_mass, mass_ratio, chi_1, chi_2, luminosity_distance, theta_jn, phase = (
            parameters["chirp_mass"],
            parameters["mass_ratio"],
            parameters["chi_1"],
            parameters["chi_2"],
            parameters["luminosity_distance"],
            parameters["theta_jn"],
            parameters["phase"],
        )
        m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)
        Time_array_lal_sequence = lal.CreateREAL8Sequence(len(Time_array))
        Time_array_lal_sequence.data = Time_array / ((m1 + m2) * lal.MTSUN_SI)
        h = ls.SimIMRPhenomTHM_neha_truncate(
            m1 * lal.MSUN_SI,
            m2 * lal.MSUN_SI,
            chi_1,
            chi_2,
            luminosity_distance * 1e6 * lal.PC_SI,
            theta_jn,
            dT,
            fmin,
            fref,
            phase,
            Time_array_lal_sequence,
            self.lalParams,
        )
        return h[0].data.data - 1j * h[1].data.data

    def log_likelihood_ratio(self):
        """
        Computes exact log likelihood.
        """
        log_like = 0
        for k in self.Detectors_list.keys():
            t_delay = self.Detectors_list[k].time_delay_from_earth_center(
                right_ascension=self.parameters["ra"],
                declination=self.parameters["dec"],
                t_gps=self.parameters["geocent_time"],
            )
            h = self.waveform(
                self.parameters,
                self.Time_array_per_detector[k]
                - (
                    t_delay
                    + self.parameters["geocent_time"]
                    - self.injection_parameters["geocent_time"]
                ),
            )

            # Antenna pattern
            F_plus, F_cross = self.Detectors_list[k].antenna_pattern(
                right_ascension=self.parameters["ra"],
                declination=self.parameters["dec"],
                polarization=self.parameters["psi"],
                t_gps=self.parameters["geocent_time"],
            )
            signal = F_plus * np.real(h) + F_cross * (-np.imag(h))
            log_like += np.dot(
                self.data_times_C_inv[k], signal
            ) - 0.5 * self.weighted_inner_product_circulant(
                self.x[k], self.y[k], signal, signal
            )
        return log_like

    def noise_log_likelihood(self):
        """
        Computes log likelihood for noise.
        """
        if self.noise_log_likelihood_value is None:
            noise_log_like = 0
            for k in self.Detectors_list.keys():
                noise_log_like += -0.5 * self.data_times_C_inv_times_data[k]
            self.noise_log_likelihood_value = noise_log_like
        return self.noise_log_likelihood_value

    def log_likelihood(self):
        """
        Computes log likelihood.
        """
        log_like = self.log_likelihood_ratio() + self.noise_log_likelihood()
        return log_like
>>>>>>> 9ea120d (Wire solver abstraction into all likelihood classes)
