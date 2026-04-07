from scipy.interpolate import interp1d
import cmath
from typing import Tuple
from helper_functions import *
import logging

logging.basicConfig(level=logging.INFO)


class RadarDataProcessing:

    def __init__(self,
                 tile_array: np.ndarray,
                 meta_dataframe: pd.DataFrame,
                 ephemeris: pd.DataFrame
                 ):

        self.tile_array = tile_array
        self.meta_dataframe = meta_dataframe
        self.ephemeris = ephemeris

    def squint_angle_calc(self) -> Tuple[np.ndarray, list, np.ndarray]:

        logging.info('############### Squint angle calculation #########################')
        c = 299792458.0
        len_range_line = self.tile_array.shape[1]
        len_az_line = self.tile_array.shape[0]

        RGDEC = self.meta_dataframe["range_decimation"].unique()[0]
        PRI = self.meta_dataframe["pri"].unique()[0]
        rank = self.meta_dataframe["rank"].unique()[0]
        suppressed_data_time = 320 / (8 * F_REF)
        range_start_time = self.meta_dataframe["sampling_window_start_time"].unique()[0] + suppressed_data_time
        wavelength = c / 5.405e9

        # Sample rates
        range_sample_freq = range_dec_to_sample_rate(RGDEC)
        range_sample_period = 1 / range_sample_freq
        az_sample_freq = 1 / PRI
        # az_sample_period = PRI

        # Fast time vector - defines the time axis along the fast time direction
        range_line_num = [i for i in range(len_range_line)]
        fast_time = []
        for i in range_line_num:
            fast_time.append(range_start_time + i * range_sample_period)

        # Slant range vector - defines R0, the range of closest approach, for each range cell
        slant_range = []
        for t in fast_time:
            slant_range.append((rank * PRI + t) * c / 2)

        # Axes - defines the frequency axes in each direction after FFT
        # SWL = len_range_line / range_sample_freq
        az_freq_vals = np.arange(-az_sample_freq / 2, az_sample_freq / 2, 1 / (PRI * len_az_line))
        # range_freq_vals = np.arange(-range_sample_freq / 2, range_sample_freq / 2, 1 / SWL)

        # Need two parameters which vary over range and azimuth
        # D is the cosine of the instantaneous squint angle and is defined by the letter D in most literature
        # Define a function to calculate D
        def d(range_freq, velocity):
            return math.sqrt(1 - ((wavelength ** 2 * range_freq ** 2) / (4 * velocity ** 2)))

        D = np.zeros((len_az_line, len_range_line))

        # Spacecraft velocity - numerical calculation of the effective spacecraft velocity
        ecef_vels = self.ephemeris.apply(lambda x: math.sqrt(
            x["X-axis velocity ECEF"] ** 2 + x["Y-axis velocity ECEF"] ** 2 + x["Z-axis velocity ECEF"] ** 2), axis=1)
        velocity_interp = interp1d(self.ephemeris["POD Solution Data Timestamp"].unique(), ecef_vels.unique(),
                                   fill_value="extrapolate")
        x_interp = interp1d(self.ephemeris["POD Solution Data Timestamp"].unique(),
                            self.ephemeris["X-axis position ECEF"].unique(),
                            fill_value="extrapolate")
        y_interp = interp1d(self.ephemeris["POD Solution Data Timestamp"].unique(),
                            self.ephemeris["Y-axis position ECEF"].unique(),
                            fill_value="extrapolate")
        z_interp = interp1d(self.ephemeris["POD Solution Data Timestamp"].unique(),
                            self.ephemeris["Z-axis position ECEF"].unique(),
                            fill_value="extrapolate")
        space_velocities = self.meta_dataframe.apply(lambda x: velocity_interp(x["coarse_time"] + x["fine_time"]),
                                                     axis=1)

        x_positions = self.meta_dataframe.apply(lambda x: x_interp(x["coarse_time"] + x["fine_time"]), axis=1).to_list()
        y_positions = self.meta_dataframe.apply(lambda x: y_interp(x["coarse_time"] + x["fine_time"]), axis=1).to_list()
        z_positions = self.meta_dataframe.apply(lambda x: z_interp(x["coarse_time"] + x["fine_time"]), axis=1).to_list()

        a = 6378137  # WGS84 semi major axis
        b = 6356752.3142  # WGS84 semi minor axis
        velocities = np.zeros((len_az_line, len_range_line))

        # loop over range and azimuth,calculate spacecraft velocity and D
        for i in range(len_az_line):
            H = math.sqrt(x_positions[i] ** 2 + y_positions[i] ** 2 + z_positions[i] ** 2)
            W = float(space_velocities.iloc[i]) / H
            lat = math.atan(z_positions[i] / x_positions[i])
            local_earth_rad = math.sqrt(((a ** 2 * math.cos(lat)) ** 2 + (b ** 2 * math.sin(lat)) ** 2) / (
                    (a * math.cos(lat)) ** 2 + (b * math.sin(lat)) ** 2))
            for j in range(len_range_line):
                cos_beta = (local_earth_rad ** 2 + H ** 2 - slant_range[j] ** 2) / (2 * local_earth_rad * H)
                this_ground_velocity = local_earth_rad * W * cos_beta
                velocities[i, j] = math.sqrt(float(space_velocities.iloc[i]) * this_ground_velocity)
                D[i, j] = d(az_freq_vals[i], velocities[i, j])

        return D, slant_range, az_freq_vals

    def conversion_time_frq_domain(self) -> np.ndarray:

        logging.info('############### conversion time to frequency domain #########################')

        len_range_line = self.tile_array.shape[1]
        len_az_line = self.tile_array.shape[0]

        freq_domain_data = np.zeros((len_az_line, len_range_line), dtype=complex)

        for az_index in range(len_az_line):
            range_line = self.tile_array[az_index, :]
            range_fft = np.fft.fft(range_line)
            freq_domain_data[az_index, :] = range_fft

        for range_index in range(len_range_line):
            az_line = freq_domain_data[:, range_index]
            az_fft = np.fft.fft(az_line)
            az_fft = np.fft.fftshift(az_fft)
            freq_domain_data[:, range_index] = az_fft

        return freq_domain_data

    def range_filter_from_replica_pulse(
            self,
            freq_domain_data: np.ndarray
    ) -> np.ndarray:

        logging.info('############### range_filter_from_replica_pulse #########################')

        len_range_line = self.tile_array.shape[1]
        len_az_line = self.tile_array.shape[0]

        TXPL = self.meta_dataframe["tx_pulse_length"].unique()[0]
        TXPSF = self.meta_dataframe["txpsf"].unique()[0]
        TXPRR = self.meta_dataframe["txprr"].unique()[0]
        RGDEC = self.meta_dataframe["range_decimation"].unique()[0]
        range_sample_freq = range_dec_to_sample_rate(RGDEC)
        num_tx_vals = int(TXPL * range_sample_freq)
        tx_replica_time_vals = np.linspace(-TXPL / 2, TXPL / 2, num=num_tx_vals)
        phi1 = TXPSF + TXPRR * TXPL / 2
        phi2 = TXPRR / 2
        tx_replica = np.zeros(num_tx_vals, dtype=complex)
        for i in range(num_tx_vals):
            tx_replica[i] = cmath.exp(
                2j * cmath.pi * (phi1 * tx_replica_time_vals[i] + phi2 * tx_replica_time_vals[i] ** 2))

        range_filter = np.zeros(len_range_line, dtype=complex)
        index_start = np.ceil((len_range_line - num_tx_vals) / 2) - 1
        index_end = num_tx_vals + np.ceil((len_range_line - num_tx_vals) / 2) - 2
        range_filter[int(index_start):int(index_end + 1)] = tx_replica

        range_filter = np.fft.fft(range_filter)
        range_filter = np.conjugate(range_filter)

        for az_index in range(len_az_line):
            freq_domain_data[az_index, :] = freq_domain_data[az_index, :] * range_filter

        return freq_domain_data

    def rcmc_filter(
            self,
            D: np.ndarray,
            slant_range: list,
            freq_domain_data: np.ndarray
    ) -> np.ndarray:

        logging.info('############### creating rcmc filter #########################')

        c = 299792458
        len_range_line = self.tile_array.shape[1]
        len_az_line = self.tile_array.shape[0]

        RGDEC = self.meta_dataframe["range_decimation"].unique()[0]
        range_sample_freq = range_dec_to_sample_rate(RGDEC)

        rcmc_filt = np.zeros(len_range_line, dtype=complex)
        range_freq_vals = np.linspace(-range_sample_freq / 2, range_sample_freq / 2, num=len_range_line)
        for az_index in range(len_az_line):
            rcmc_filt = np.zeros(len_range_line, dtype=complex)
            for range_index in range(len_range_line):
                rcmc_shift = slant_range[0] * ((1 / D[az_index, range_index]) - 1)
                rcmc_filt[range_index] = cmath.exp(4j * cmath.pi * range_freq_vals[range_index] * rcmc_shift / c)
            freq_domain_data[az_index, :] = freq_domain_data[az_index, :] * rcmc_filt

        return rcmc_filt

    def doppler_correction(
            self,
            freq_domain_data: np.ndarray
    ):

        logging.info('############### doppler correction #########################')

        len_range_line = self.tile_array.shape[1]
        len_az_line = self.tile_array.shape[0]

        range_doppler_data = np.zeros((len_az_line, len_range_line), dtype=complex)
        for range_line_index in range(len_az_line):
            ifft = np.fft.ifft(freq_domain_data[range_line_index, :])
            ifft_sorted = np.fft.ifftshift(ifft)
            range_doppler_data[range_line_index, :] = ifft_sorted

        return range_doppler_data

    def azimuth_compression(
            self,
            D: np.ndarray,
            range_doppler_data: np.ndarray,
            az_freq_vals: np.ndarray,
            slant_range: list
    ) -> np.ndarray:

        logging.info('############### azimuth compression data generation #########################')

        c = 299792458
        wavelength = c / 5.405e9
        len_range_line = self.tile_array.shape[1]
        len_az_line = self.tile_array.shape[0]
        # Create azimuth filter
        az_compressed_data = np.zeros((len_az_line, len_range_line), 'complex')

        for az_line_index in range(len_range_line):
            # d_vector = np.zeros(len_az_line)
            this_az_filter = np.zeros(len_az_line, 'complex')
            for i in range(len(az_freq_vals)):
                this_az_filter[i] = cmath.exp(4j * cmath.pi * slant_range[i] * D[i, az_line_index] / wavelength)
            result = range_doppler_data[:, az_line_index] * this_az_filter[:]
            result = np.fft.ifft(result)
            az_compressed_data[:, az_line_index] = result

        return az_compressed_data
		
		# some functionalities of code are inspired from these github pages:
#https://github.com/Rich-Hall/sentinel1decoder
#https://nbviewer.org/github/Rich-Hall/sentinel1Level0DecodingDemo/blob/main/sentinel1Level0DecodingDemo.ipynb
