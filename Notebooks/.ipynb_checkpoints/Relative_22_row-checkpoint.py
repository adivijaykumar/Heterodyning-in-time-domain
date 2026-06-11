#!/usr/bin/env python
# coding: utf-8
import sys
import bilby
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
import numpy as np
import lalsimulation as ls
import lal
from scipy.optimize import differential_evolution

print(ls.__file__)


class RelativeBinningTimeDomain_H1detectorframe_for_longer_signals(bilby.Likelihood):
    """
    Parameters
    ----------
    time : 1D array
        An array of time points at which waveform/data is generated.
    Detectors_list : list
        List of detector names (pycbc.detector.Detector)
    fiducial_parameters : dict
        Dictionary of fiducial parameters.
    injection_parameters : dict
        Dictionary of injection parameters.
    Data_list : list
        List containing the observed data from each detector.
    C_inv_row : array-like
        1st row of the inverse covariance matrix.
    Noise : array-like
        Noise properties of the detectors.
    epsilon_array_per_detector : dict, optional (default={'H1': np.array([0.5, 0.1]), 'L1': np.array([0.5, 0.1]), 'V1': np.array([0.5, 0.1])})
        A tolerance value used to determine the size of time bins.
        The smaller value of epsilon generates bins of smaller size, resulting in more number of bins.
        If epsilon is too small, then the size of bins may be smaller than the sampling frequency.
        We require that each bin contain atleast one time point.
    spacing : list, optional (default=[1, 10, 100])
        Number of time points in each time bin (only for merger and ringdown part for 0.0 < self.time < 0.1)
    spacing_times : array-like, optional (default = np.array([0.02, 0.1]))
        Time points at which the spacing changes.    
    time_split_per_detector : dict, optional (default = {"H1": -0.02, "L1": -0.02, "V1": -0.1})
        Time points at which the epsilon_array changes for each detector.
    priors : dict, optional
        Prior distributions for the model parameters.
    
    Notes
    -----
    This class uses IMRPhenomT waveform approximant for generating waveforms.
    """

    def __init__(
        self,
        time,
        Detectors_list,
        fiducial_parameters,
        injection_parameters,
        Data_list,
        C_inv_row,
        Noise,
        trigger_time,
        detector_frame = "H1",
        epsilon_array_per_detector={
            "H1": np.array([0.05, 0.01]),
            "L1": np.array([0.05, 0.01]),
            "V1": np.array([0.05, 0.01]),
        },
        spacing=np.array([1, 10, 100]),
        spacing_times=np.array([0.02, 0.1]),
        time_split_per_detector={"H1": -0.02, "L1": -0.02, "V1": -0.1},
        priors=None,
    ):
        super().__init__(parameters={})
        self.gamma = np.array([5 / 8, 3 / 8, 1 / 4, -5 / 8, -3 / 8]) # [5 / 8, 3 / 8, 1 / 4, -3 / 8]
        self.chi = 1
        self.time = time
        self.epsilon_array_per_detector = epsilon_array_per_detector
        self.spacing = spacing
        self.spacing_times = spacing_times
        self.time_split_per_detector = time_split_per_detector
        self.Detectors_list = Detectors_list
        self.Noise = Noise
        if fiducial_parameters is None:
            fiducial_parameters = priors.sample()
        self.fiducial_parameters = fiducial_parameters.copy()
        if injection_parameters is None:
            self.injection_parameters = None
        else:
            self.injection_parameters = injection_parameters.copy()
        self.C_inv_row = C_inv_row
        self.noise_log_likelihood_value = None
        self._Data_list = Data_list if Data_list else {}
        self.detector_frame = detector_frame
        self.trigger_time = trigger_time

        self.compute_SNR_TD_and_waveform_data()
        self.setup_bins_per_detector()
        self.Summary_data()

    @property
    def injection_parameters(self):
        return self._injection_parameters

    @injection_parameters.setter
    def injection_parameters(self, new_params):
        if new_params is None:
            self._injection_parameters = None
        else:
            self._injection_parameters = new_params.copy()
        self._Data_list = {}

    @property
    def Data_list(self):
        return self._Data_list

    def inner_product_TD(self, h1, h2, InvCov):
        """
        Computes inner product between h1, InvCov and h2.

        Parameters
        ----------
        h1 : 1D array
            Data or waveform
        h2 : 1D array
            Data or waveform
        InvCov : Ndarray
            Inverse of n-dimensional covariance matrix

        Returns
        -------
        Inner product : float
            Inner product between h1, InvCov, h2, calculated as
            Sum_{ij} h1_i C^{-1}_{ij} h2_j
        """
        return np.dot(h1, np.matmul(InvCov, h2))

    def weighted_inner_product_circulant(self, h1, h2, C_inv_row):
        """
        Compute (h1, C^-1 h2) using FFT
        """
        C_inv_h2 = self.vector_circulant_matrix_multiply(h2, C_inv_row)
        return np.dot(h1, C_inv_h2)

    def vector_circulant_matrix_multiply(self, h, C_inv_row):
        """
        Compute C^-1 h using FFT
        """
        fft_C = np.fft.fft(C_inv_row)
        fft_h = np.fft.fft(h)
        result = np.fft.ifft(fft_C * fft_h).real
        return result

    def compute_SNR_TD_and_waveform_data(self):
        """
        Computes fiducial waveform (or 22 mode), total SNR and inner product between data, inverse-covariance matrix and data.
        """
        data_times_C_inv = {}
        Antenna_pattern_dict = {}
        data_times_C_inv_times_data = {}
        # Generating 22-mode using fiducial parameters. We require this to compute summary data and likelihood.
        h22_fiducial = {}
        # Generating plus and cross polarizations to compute SNR and data.
        h_injection = {}
        Time_array_per_detector = {}
        t_delay_injection_dict = {}

        for k in self.Detectors_list.keys():
            t_delay_fiducial = self.Detectors_list[k].time_delay_from_detector(
                other_detector=self.Detectors_list[self.detector_frame],
                right_ascension=self.fiducial_parameters["ra"],
                declination=self.fiducial_parameters["dec"],
                t_gps=self.fiducial_parameters[f"{self.detector_frame}_time"],
            )
            Time_array_per_detector[k] = self.time
            h22_fiducial[k] = self.waveform_22_modes_fiducial(
                self.fiducial_parameters, Time_array_per_detector[k] - (t_delay_fiducial+ self.fiducial_parameters[f"{self.detector_frame}_time"] - self.trigger_time)
            )
            if self.injection_parameters is not None:
                h_injection[k] = self.waveform(
                    self.injection_parameters,
                    Time_array_per_detector[k] - t_delay_injection,
                )
                t_delay_injection = self.Detectors_list[k].time_delay_from_detector(
                    other_detector=self.Detectors_list[self.detector_frame],
                    right_ascension=self.injection_parameters["ra"],
                    declination=self.injection_parameters["dec"],
                    t_gps=self.injection_parameters[f"{self.detector_frame}_time"],
                )

                F_plus, F_cross = self.Detectors_list[k].antenna_pattern(
                    right_ascension=self.injection_parameters["ra"],
                    declination=self.injection_parameters["dec"],
                    polarization=self.injection_parameters["psi"],
                    t_gps=self.injection_parameters[f"{self.detector_frame}_time"],
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
            data_times_C_inv[k] = self.vector_circulant_matrix_multiply(
                self.Data_list[k], self.C_inv_row[k]
            )
            data_times_C_inv_times_data[k] = self.weighted_inner_product_circulant(
                self.Data_list[k], self.Data_list[k], self.C_inv_row[k]
            )
        self.data_times_C_inv_times_data = data_times_C_inv_times_data
        self.Time_array_per_detector = Time_array_per_detector
        self.data_times_C_inv = data_times_C_inv
        self.h22_fiducial = h22_fiducial
        if self.injection_parameters is not None:
            self.h_injection = h_injection
            self.t_delay_injection_dict = t_delay_injection_dict

        if self.injection_parameters is not None:
            SNR_dict = {}
            SNR_arr = []
            for k in self.Detectors_list.keys():
                signal = Antenna_pattern_dict[k]["F_plus"] * np.real(
                    self.h_injection[k]
                ) + Antenna_pattern_dict[k]["F_cross"] * (-np.imag(self.h_injection[k]))
                hh = self.weighted_inner_product_circulant(
                    signal, signal, self.C_inv_row[k]
                )
                dh = self.weighted_inner_product_circulant(
                    self.Data_list[k], signal, self.C_inv_row[k]
                )
                SNR = np.abs(dh) / np.sqrt(np.abs(hh))
                SNR_arr.append(SNR)
                SNR_dict[k] = SNR
            self.SNR_dict = SNR_dict
            self.network_SNR = np.sqrt(np.sum(np.array(SNR_arr) ** 2))

        if self.injection_parameters is None:
            self.SNR_dict = None
            self.network_SNR = None
            self.h_injection = None
        return (
            self.SNR_dict,
            self.network_SNR,
            self.h_injection,
            self.Data_list,
            self.data_times_C_inv,
        )

    # def full_waveform(self, h1, ho, TimeArray):
    #     """
    #     Computes target waveform (or mode (h^{lm})) at each time point using the ratio of target waveform (or mode) generated at bin edges to the fiducial waveform (or mode).

    #     Parameters
    #     ----------
    #     h1 : 1D array
    #         Target waveform or mode, generated only at bin edges
    #     ho : 1D array
    #         Fiducial waveform or mode, generated at each time point

    #     Returns
    #     -------
    #     h_full : 1D array
    #         Target waveform at each time point, calculated as
    #         h(t) = r(t) ho(t)  OR  h^{lm}(t) = r^{lm}(t) ho^{lm}(t)
    #     """
    #     waveform_ratio = h1 / ho[self.bin_edge_index]

    #     r_o = (waveform_ratio[1:] + waveform_ratio[:-1]) / 2
    #     r_1 = (waveform_ratio[1:] - waveform_ratio[:-1]) / self.bin_width

    #     h_ratio = np.zeros(len(TimeArray), dtype="complex")

    #     for i in range(self.number_of_bins):
    #         start_index = self.bin_edge_index[i]
    #         end_index = self.bin_edge_index[i + 1]
    #         index = np.arange(start_index, end_index + 1, 1)
    #         time_per_bin = TimeArray[index]
    #         h_ratio[index] = r_o[i] + r_1[i] * (time_per_bin - self.bin_centre[i])
    #     h_full = h_ratio * ho
    #     return h_full

    def waveform(self, parameters, Time_array, fmin=20, fref=20, dT=1 / 4096.0):
        """
        Computes time-domain waveform (approximant - IMRPhenomT).

        Parameters
        ----------
        parameters : dict
            A dictionary containing the parameters of binary system.
        time : 1D array
            A numpy array of time points at which the h22 mode is to be computed.

        Returns
        -------
        h : 1D array
            time-domain waveform
        """
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
        h = ls.SimIMRPhenomT_neha_truncate(
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
            None,
        )
        return h[0].data.data - 1j * h[1].data.data

    def waveform_22_modes(
        self, parameters, Time_array, fmin=20, fref=20, dT=1 / 4096.0
    ):
        """
        Computes time-domain 22 mode (h22).
        (approximant - IMRPhenomT)

        Parameters
        ----------
        parameters : dict
            A dictionary containing the parameters of binary system.
        time : 1D array
            A numpy array of time points at which the h22 mode is to be computed.

        Returns
        -------
        h22 : 1D array
            time-domain 22 mode (h22)
        """
        chirp_mass, mass_ratio, chi_1, chi_2, luminosity_distance, phase = (
            parameters["chirp_mass"],
            parameters["mass_ratio"],
            parameters["chi_1"],
            parameters["chi_2"],
            parameters["luminosity_distance"],
            parameters["phase"],
        )
        m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)
        Time_array_lal_sequence = lal.CreateREAL8Sequence(len(Time_array))
        Time_array_lal_sequence.data = Time_array / ((m1 + m2) * lal.MTSUN_SI)
        h = ls.SimIMRPhenomT_neha_just_modes_truncate(
            m1 * lal.MSUN_SI,
            m2 * lal.MSUN_SI,
            chi_1,
            chi_2,
            luminosity_distance * 1e6 * lal.PC_SI,
            dT,
            fmin,
            fref,
            phase,
            Time_array_lal_sequence,
            None,
        )
        return h.mode.data.data

    
    def waveform_22_modes_fiducial(
        self, parameters, Time_array, fmin=20, fref=20, dT=1 / 4096.0
    ):
        """
        Computes time-domain 22 mode (h22).
        (approximant - IMRPhenomT)

        Parameters
        ----------
        parameters : dict
            A dictionary containing the parameters of binary system.
        time : 1D array
            A numpy array of time points at which the h22 mode is to be computed.

        Returns
        -------
        h22 : 1D array
            time-domain 22 mode (h22)
        """
        chirp_mass, mass_ratio, chi_1, chi_2, luminosity_distance, phase = (
            parameters["chirp_mass"],
            parameters["mass_ratio"],
            parameters["chi_1"],
            parameters["chi_2"],
            parameters["luminosity_distance"],
            parameters["phase"],
        )
        m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)
        Time_array_lal_sequence = lal.CreateREAL8Sequence(len(Time_array))
        Time_array_lal_sequence.data = Time_array / ((m1 + m2) * lal.MTSUN_SI)
        h = ls.SimIMRPhenomT_neha_just_modes(
            m1 * lal.MSUN_SI,
            m2 * lal.MSUN_SI,
            chi_1,
            chi_2,
            luminosity_distance * 1e6 * lal.PC_SI,
            dT,
            fmin,
            fref,
            phase,
            Time_array_lal_sequence,
            None,
        )
        h22 = h.mode.data.data
        # h22[Time_array>0.02] = 1e-22 + 1j*1e-22
        return h22

    def setup_bins_inspiral(self, Time_array, time_split, epsilon_array):
        """
        Constructs non-uniform time bins for inspiral part.
        The method for bin construction is similar to the one presented in
        https://arxiv.org/abs/1806.08792.

        Returns
        -------
        number_of_bins : float
            Total number of bins constructed for inspiral part
        bin_edge_index : 1D array
            Indices corresponding to the edge of time bins
        bin_centre : 1D array
            Centre of each time bin
        bin_width : 1D array
            Width of each time bin
        time_on_edge : 1D array
            The time value at each bin edge

        Variables
        ----------
        epsilon : float
            A tolerance value used to determine the size of time bins.
            The smaller value of epsilon generates bins of smaller size, resulting in more number of bins.
            If epsilon is too small, then the size of bins may be smaller than the sampling frequency.
            We require that each bin contain atleast one time point.
        """

        d_alpha = []
        for g in self.gamma:
            if g < 0:
                d_alpha.append(2 * np.pi * self.chi / (min(abs(Time_array))) ** g)
            else:
                d_alpha.append(2 * np.pi * self.chi / (max(abs(Time_array))) ** g)
        d_alpha = np.array(d_alpha)

        d_phi_f = np.sum(
            np.sign(self.gamma) * d_alpha * (abs(Time_array[:, None]) ** self.gamma),
            axis=1,
        )

        mask1 = Time_array <= time_split
        mask2 = Time_array > time_split
        d_phi_f_regions = [d_phi_f[mask1], d_phi_f[mask2]]
        idx_offsets = [0, np.sum(mask1)]

        bin_edge_indices = []
        for region_idx, (d_phi_f_reg, e) in enumerate(
            zip(d_phi_f_regions, epsilon_array)
        ):
            number_of_bins = int(abs((d_phi_f_reg[-1] - d_phi_f_reg[0]) / e))

            region_bin_edges = []
            for i in range(number_of_bins + 1):
                bin_idx = np.where(
                    d_phi_f_reg - d_phi_f_reg[0]
                    >= (i / number_of_bins) * (d_phi_f_reg[-1] - d_phi_f_reg[0])
                )[0]
                if len(bin_idx) > 0:
                    region_bin_edges.append(bin_idx[-1])
                else:
                    region_bin_edges.append(0)

            region_bin_edges = np.array(region_bin_edges) + idx_offsets[region_idx]
            bin_edge_indices.extend(region_bin_edges)

        bin_edge_index = np.unique(np.array(bin_edge_indices))
        time_on_edge = Time_array[bin_edge_index]
        bin_width = time_on_edge[1:] - time_on_edge[:-1]
        bin_centre = (time_on_edge[1:] + time_on_edge[:-1]) / 2
        number_of_bins = len(bin_centre)
        return (number_of_bins, bin_edge_index, bin_centre, bin_width, time_on_edge)

    def setup_bins(self, Time_array_full, time_split, epsilon_array):
        """
        Constructs time bins for entire time range.

        Returns
        -------
        number_of_bins : float
            Total number of bins created for entire time range
        bin_edge_index : 1D array
            Indices corresponding to the edge of time bins
        bin_centre : 1D array
            Centre of each time bin
        bin_width : 1D array
            Width of each time bin
        time_on_edge : 1D array
            The time value at each bin edge

        Variables
        ---------
        spacing : float
            Number of time points in each time bin (only for merger and ringdown part for 0.0 < self.time < 0.1)
        """
        idx_inspiral_end = np.where(Time_array_full < -0.0)[0][-1]
        Time_array_inspiral = Time_array_full[: idx_inspiral_end + 1]
        bin_edge_index_insp = self.setup_bins_inspiral(
            Time_array_inspiral, time_split, epsilon_array
        )[1]

        # We construct uniform time bins for merger and ringdwon part.

        spacing_t_less_t1 = self.spacing[0]  # Spacing for t < t1
        spacing_t_t1_to_t2 = self.spacing[1]  # Spacing for t1 =< t < t2
        spacing_t_greater_t2 = self.spacing[2]  # Spacing for t >= t2

        idx_merger_ringdown_start = idx_inspiral_end
        if Time_array_full[-1] > self.spacing_times[0]:
            idx_t_t1 = np.where(Time_array_full > self.spacing_times[0])[0][0]
        else:
            idx_t_t1 = int(len(Time_array_full))
        if Time_array_full[-1] > self.spacing_times[1]:
            idx_t_t2 = np.where(Time_array_full > self.spacing_times[1])[0][0]
        else:
            idx_t_t2 = int(len(Time_array_full))

        # Define bins in the three different regions : t < t1, t = t1 to t2 and t > t2
        bins_t_less_t1 = np.arange(
            idx_merger_ringdown_start, idx_t_t1, spacing_t_less_t1
        )
        if Time_array_full[-1] > self.spacing_times[0]:
            bins_t_t1_to_t2 = np.arange(idx_t_t1, idx_t_t2, spacing_t_t1_to_t2)
        else:
            bins_t_t1_to_t2 = np.array([], dtype=int)
        if Time_array_full[-1] > self.spacing_times[1]:
            bins_t_greater_t2 = np.arange(
                idx_t_t2, len(Time_array_full), spacing_t_greater_t2
            )
        else:
            bins_t_greater_t2 = np.array([], dtype=int)
        # Combine all bins
        bin_edge_index = np.concatenate(
            (
                bin_edge_index_insp,
                bins_t_less_t1[1:],
                bins_t_t1_to_t2[1:],
                bins_t_greater_t2[1:],
            ),
            dtype=int,
        )

        if bin_edge_index[-1] != len(Time_array_full) - 1:
            bin_edge_index = np.insert(
                bin_edge_index, len(bin_edge_index), len(Time_array_full) - 1
            )
        time_on_edge = Time_array_full[bin_edge_index]
        bin_width = time_on_edge[1:] - time_on_edge[:-1]
        bin_centre = (time_on_edge[1:] + time_on_edge[:-1]) / 2
        number_of_bins = len(bin_centre)
        return (time_on_edge, number_of_bins, bin_edge_index, bin_centre, bin_width)

    def setup_bins_per_detector(self):
        (
            time_on_edge_per_detector,
            number_of_bins_per_detector,
            bin_edge_index_per_detector,
            bin_centre_per_detector,
            bin_width_per_detector,
        ) = ({}, {}, {}, {}, {})
        for k in self.Detectors_list.keys():
            time_on_edge, number_of_bins, bin_edge_index, bin_centre, bin_width = self.setup_bins(
                self.Time_array_per_detector[self.detector_frame],
                self.time_split_per_detector[k],
                self.epsilon_array_per_detector[k],
            )
            time_on_edge_per_detector[k] = (
                time_on_edge.copy() #+ self.t_delay_injection_dict[k]
            )
            number_of_bins_per_detector[k] = number_of_bins
            bin_edge_index_per_detector[k] = bin_edge_index.copy()
            bin_centre_per_detector[k] = (
                bin_centre.copy() #+ self.t_delay_injection_dict[k]
            )
            bin_width_per_detector[k] = bin_width.copy()

        self.time_on_edge = time_on_edge_per_detector
        self.number_of_bins = number_of_bins_per_detector
        self.bin_edge_index = bin_edge_index_per_detector
        self.bin_centre = bin_centre_per_detector
        self.bin_width = bin_width_per_detector

    def Summary_data(self):
        """
        Computes summary data using FFT-based convolution for efficient B-matrix computation.
        """
        Summary_data_dict = {}

        for k in self.Detectors_list.keys():
            time_diff_arr = np.zeros_like(self.Time_array_per_detector[k], dtype=float)
            for i in range(self.number_of_bins[k]):
                start_index = self.bin_edge_index[k][i]
                end_index = self.bin_edge_index[k][i + 1]
                time_diff_arr[start_index:end_index] = (
                    self.Time_array_per_detector[k][start_index:end_index]
                    - self.bin_centre[k][i]
                )
            h = self.h22_fiducial[k]
            h_t = h * time_diff_arr
            h_conj = np.conjugate(h)
            C_inv = self.C_inv_row[k]
            fft_C = np.fft.fft(C_inv)

            data_times_C_inv_times_h22 = self.data_times_C_inv[k] * h
            data_times_C_inv_times_h22_times_time_diff = (
                data_times_C_inv_times_h22 * time_diff_arr
            )

            A_0 = np.zeros(self.number_of_bins[k], dtype=complex)
            A_1 = np.zeros(self.number_of_bins[k], dtype=complex)

            for b1 in range(self.number_of_bins[k]):
                start_index = self.bin_edge_index[k][b1]
                if b1 == self.number_of_bins[k] - 1:
                    # If last bin, include the last index in the bin
                    end_index = self.bin_edge_index[k][b1 + 1] + 1
                else:
                    # Else, do not include the last index in the bin
                    end_index = self.bin_edge_index[k][b1 + 1]

                A_0[b1] = np.sum(data_times_C_inv_times_h22[start_index:end_index])
                A_1[b1] = np.sum(
                    data_times_C_inv_times_h22_times_time_diff[start_index:end_index]
                )

            B_0 = np.zeros(
                (self.number_of_bins[k], self.number_of_bins[k]), dtype=complex
            )
            B_1 = np.zeros(
                (self.number_of_bins[k], self.number_of_bins[k]), dtype=complex
            )
            B_2 = np.zeros(
                (self.number_of_bins[k], self.number_of_bins[k]), dtype=complex
            )
            B_3 = np.zeros(
                (self.number_of_bins[k], self.number_of_bins[k]), dtype=complex
            )

            for b2 in range(self.number_of_bins[k]):
                start_index_b2 = self.bin_edge_index[k][b2]
                if b2 == self.number_of_bins[k] - 1:
                    # If last bin, include the last index in the bin
                    end_index_b2 = self.bin_edge_index[k][b2 + 1] + 1
                else:
                    # Else, do not include the last index in the bin
                    end_index_b2 = self.bin_edge_index[k][b2 + 1]

                h_bin = np.zeros_like(h)
                h_bin_t = np.zeros_like(h_t)
                h_bin[start_index_b2:end_index_b2] = h[start_index_b2:end_index_b2]
                h_bin_t[start_index_b2:end_index_b2] = h_t[start_index_b2:end_index_b2]
                fft_h = np.fft.fft(h_bin)
                fft_h_t = np.fft.fft(h_bin_t)
                conv1 = np.fft.ifft(fft_C * fft_h)
                conv2 = np.fft.ifft(fft_C * fft_h_t)

                for b1 in range(self.number_of_bins[k]):
                    start_index_b1 = self.bin_edge_index[k][b1]
                    if b1 == self.number_of_bins[k] - 1:
                        # If last bin, include the last index in the bin
                        end_index_b1 = self.bin_edge_index[k][b1 + 1] + 1
                    else:
                        # Else, do not include the last index in the bin
                        end_index_b1 = self.bin_edge_index[k][b1 + 1]

                    B_0[b1, b2] = np.dot(
                        h[start_index_b1:end_index_b1],
                        conv1[start_index_b1:end_index_b1],
                    )
                    B_1[b1, b2] = np.dot(
                        h[start_index_b1:end_index_b1],
                        conv2[start_index_b1:end_index_b1],
                    )
                    B_2[b1, b2] = np.dot(
                        h_conj[start_index_b1:end_index_b1],
                        conv1[start_index_b1:end_index_b1],
                    )
                    B_3[b1, b2] = np.dot(
                        h_conj[start_index_b1:end_index_b1],
                        conv2[start_index_b1:end_index_b1],
                    )
            Summary_data_dict[k] = (A_0, A_1, B_0, B_1, B_2, B_3)
        self.Summary_data_dict = Summary_data_dict

    def SphericalHarmonics22(self, theta, phi, m):
        return (
            np.sqrt(5.0 / (64.0 * np.pi))
            * (1.0 + np.cos(theta))
            * (1.0 + np.cos(theta))
            * np.exp(1j * m * phi)
        )

    def SphericalHarmonics2minus2(self, theta, phi, m):
        return (
            np.sqrt(5.0 / (64.0 * np.pi))
            * (1.0 - np.cos(theta))
            * (1.0 - np.cos(theta))
            * np.exp(1j * m * phi)
        )

    def log_likelihood_ratio(self):
        """
        Computes Relative Binning log likelihood using summary data and ratio of target 22 mode to fiducial 22 mode r^{22}(t).
        """
        log_like = 0
        Y_22 = self.SphericalHarmonics22(
            self.parameters["theta_jn"], np.pi / 2 - self.parameters["phase"], 2
        )
        Y_2_minus_2 = self.SphericalHarmonics2minus2(
            self.parameters["theta_jn"], np.pi / 2 - self.parameters["phase"], -2
        )
        for k in self.Detectors_list.keys():
            # Summary data
            A_0, A_1, B_0, B_1, B_2, B_3 = self.Summary_data_dict[k]

            # Time delay and 22 mode
            t_delay = self.Detectors_list[k].time_delay_from_detector(
                other_detector=self.Detectors_list[self.detector_frame],
                right_ascension=self.parameters["ra"],
                declination=self.parameters["dec"],
                t_gps=self.parameters[f"{self.detector_frame}_time"],
            )
            h22 = self.waveform_22_modes(
                self.parameters,
                self.time_on_edge[k]
                - (
                    t_delay
                    + self.parameters[f"{self.detector_frame}_time"]
                    - self.trigger_time
                ),
            )

            # Antenna pattern
            F_plus, F_cross = self.Detectors_list[k].antenna_pattern(
                right_ascension=self.parameters["ra"],
                declination=self.parameters["dec"],
                polarization=self.parameters["psi"],
                t_gps=self.parameters[f"{self.detector_frame}_time"],
            )
            h22[self.time_on_edge[k]>0.01] = 0.0 + 1j*0.0

            waveform_ratio = h22 / self.h22_fiducial[k][self.bin_edge_index[k]]
            r_0 = (waveform_ratio[1:] + waveform_ratio[:-1]) / 2
            r_1 = (waveform_ratio[1:] - waveform_ratio[:-1]) / self.bin_width[k]

            # Inner products

            # Inner_product_d_h22 = np.sum(r_0 * self.A_0 + r_1 * self.A_1)

            # Perform one fused operation: element-wise multiply and sum over all elements
            Inner_product_d_h22 = np.dot(r_0, A_0) + np.dot(r_1, A_1)

            Y_22_Y_2_minus_2_d_h22 = (
                Y_22 * Inner_product_d_h22
                + Y_2_minus_2 * np.conjugate(Inner_product_d_h22)
            )

            Inner_product_di_sj = F_plus * np.real(
                Y_22_Y_2_minus_2_d_h22
            ) - F_cross * np.imag(Y_22_Y_2_minus_2_d_h22)

            B0_r0 = np.dot(B_0, r_0)
            B1_r1 = np.dot(B_1, r_1)
            B1_r0 = np.dot(B_1, r_0)

            # Inner product: h22 with h22
            Inner_product_h22_h22 = np.dot(r_0, (B0_r0 + B1_r1)) + np.dot(r_1, B1_r0)

            # Precompute conjugated matrix-vector products
            B2_r0 = np.dot(B_2, r_0)
            B3_r1 = np.dot(B_3, r_1)
            B3_r0 = np.dot(B_3, r_0)

            # Inner product: h22 with conjugate(h22)
            Inner_product_h22_conj_h22 = np.dot(
                np.conjugate(r_0), (B2_r0 + B3_r1)
            ) + np.dot(np.conjugate(r_1), B3_r0)

            # Inner_product_h22_h22 = r_0 @ (B_0 @ r_0 + B_1 @ r_1) + r_1 @ B_1 @ r_0
            # Inner_product_h22_conj_h22 = np.conjugate(r_0) @ (B_2 @ r_0 + B_3 @ r_1) + np.conjugate(r_1) @ B_3 @ r_0

            Inner_product_sum_h22_sum_h22 = (
                Y_22 ** 2 * Inner_product_h22_h22
                + 2
                * Y_22
                * Y_2_minus_2
                * Inner_product_h22_conj_h22  # Check if this is indeed true.
                + Y_2_minus_2 ** 2 * np.conjugate(Inner_product_h22_h22)
            )

            Inner_product_sum_h22_conj_sum_h22 = np.conjugate(Y_22) * (
                Y_22 * Inner_product_h22_conj_h22
                + Y_2_minus_2 * np.conjugate(Inner_product_h22_h22)
            ) + Y_2_minus_2 * (
                Y_22 * Inner_product_h22_h22
                + np.conjugate(Y_2_minus_2) * Inner_product_h22_conj_h22
            )

            Inner_product_si_sj = (
                0.5
                * (F_plus ** 2 + F_cross ** 2)
                * np.real(Inner_product_sum_h22_conj_sum_h22)
                + 0.5
                * (F_plus ** 2 - F_cross ** 2)
                * np.real(Inner_product_sum_h22_sum_h22)
                - F_plus
                * F_cross
                * np.imag(
                    Inner_product_sum_h22_sum_h22 + Inner_product_sum_h22_conj_sum_h22
                )
            )

            log_like += Inner_product_di_sj - 0.5 * Inner_product_si_sj

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
