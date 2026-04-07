import os.path
import numpy as np
import pandas as pd
import time
import math
import struct
import joblib
from pandas import DataFrame
from lookup_table import *
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)


def save_radar_data(
        radar_data: np.ndarray,
        save_radar_data_address: str
) -> str:
    logging.info('############### Saving Radar Data #####################')
    logging.info('#################### Data Saved: {} #####################'.format(save_radar_data_address))
    np.save(os.path.join(save_radar_data_address, 'radar_data.npy'), radar_data)
    print(f'The raw data is saved at: {save_radar_data_address}')

    return save_radar_data_address


def read_dataframe(
        file_address: str,
        file_name: str
) -> pd.DataFrame:

    return pd.read_csv(os.path.join(file_address, f'{file_name}.csv'))


def read_saved_radar_data(
        saved_radar_data_path: str,
        filename: str
) -> np.ndarray:

    array = np.load(os.path.join(saved_radar_data_path, f'{filename}.npy'))
    logging.info(f'############# array shape : {array.shape} #########################')
    return np.load(os.path.join(saved_radar_data_path, f'{filename}.npy'))


def tiles_of_data(
    radar_data: np.ndarray,
    start_limit: int,
    end_limit: int
) -> np.ndarray:

    return radar_data[start_limit:end_limit]

def save_dictionary(dict,
                    fileaddress:str,
                    filename:str):
    # dumpy the fileaddress and names
    joblib.dump(dict, f'{fileaddress}\\{filename}.joblib')

def save_chunks_of_data(
        radar_data: np.ndarray,
        save_chunks_path: str
):
    for i in range(len(radar_data)):
        start_index = i * 5000
        end_index = min((i + 1) * 5000, len(radar_data))
        os.makedirs(os.path.join(save_chunks_path, 'chunks_of_data'),exist_ok=True)
        arr = radar_data[start_index:end_index]
        np.save(os.path.join(save_chunks_path, 'chunks_of_data', f'{i+1}.npy'), arr)

    print('Chunks of Data saved at :', save_chunks_path)
    return os.path.join(save_chunks_path,'chunks_of_data')

def range_dec_to_sample_rate(rgdec_code:int) -> float:
    scaling_factors = {
        0: 3,
        1: 8 / 3,
        3: 20 / 9,
        4: 16 / 9,
        5: 3 / 2,
        6: 4 / 3,
        7: 2 / 3,
        8: 12 / 7,
        9: 5 / 4,
        10: 6 / 13,
        11: 16 / 11
    }

    if rgdec_code in scaling_factors:
        return scaling_factors[rgdec_code] * F_REF
    else:
        raise Exception(f"Invalid range decimation code {rgdec_code} supplied - valid codes are {list(scaling_factors.keys())}")

def read_subcommed_data(df):
    index_col = "subcom_data_word_ind"
    data_col = "subcom_data_word"
    start_indices = df.index[df[index_col] == 1]
    output_dict_list = []

    dbl_type = np.dtype(np.float64).newbyteorder('>')
    sgl_type = np.dtype(np.float32).newbyteorder('>')
    for i in tqdm(start_indices):

        # Check that a continuous block of 64 follows our index.
        if len(df) - i >= 64:
            if all(df.loc[i:i + 63][index_col] == list(range(1, 65))):
                d = df.loc[i:i + 63][data_col].tolist()
                x_bytes = struct.pack('>HHHH', d[0], d[1], d[2], d[3])
                y_bytes = struct.pack('>HHHH', d[4], d[5], d[6], d[7])
                z_bytes = struct.pack('>HHHH', d[8], d[9], d[10], d[11])
                x = np.frombuffer(x_bytes, dtype=dbl_type)[0]
                y = np.frombuffer(y_bytes, dtype=dbl_type)[0]
                z = np.frombuffer(z_bytes, dtype=dbl_type)[0]

                x_vel_bytes = struct.pack('>HH', d[12], d[13])
                y_vel_bytes = struct.pack('>HH', d[14], d[15])
                z_vel_bytes = struct.pack('>HH', d[16], d[17])
                x_vel = np.frombuffer(x_vel_bytes, dtype=sgl_type)[0]
                y_vel = np.frombuffer(y_vel_bytes, dtype=sgl_type)[0]
                z_vel = np.frombuffer(z_vel_bytes, dtype=sgl_type)[0]

                pvt_t1 = d[18] * 2 ** 24
                pvt_t2 = (d[19] * 2 ** 8)
                pvt_t3 = (d[20] * 2 ** -8)
                pvt_t4 = (d[21] * 2 ** -24)
                pvt_t = pvt_t1 + pvt_t2 + pvt_t3 + pvt_t4

                output_dictionary = {
                    "X-axis position ECEF": x,
                    "Y-axis position ECEF": y,
                    "Z-axis position ECEF": z,
                    "X-axis velocity ECEF": x_vel,
                    "Y-axis velocity ECEF": y_vel,
                    "Z-axis velocity ECEF": z_vel,
                    "POD Solution Data Timestamp": pvt_t
                }

                q0_bytes = struct.pack('>HH', d[22], d[23])
                q1_bytes = struct.pack('>HH', d[24], d[25])
                q2_bytes = struct.pack('>HH', d[26], d[27])
                q3_bytes = struct.pack('>HH', d[28], d[29])
                q0 = np.frombuffer(q0_bytes, dtype=sgl_type)[0]
                q1 = np.frombuffer(q1_bytes, dtype=sgl_type)[0]
                q2 = np.frombuffer(q2_bytes, dtype=sgl_type)[0]
                q3 = np.frombuffer(q3_bytes, dtype=sgl_type)[0]

                x_ang_rate_bytes = struct.pack('>HH', d[30], d[31])
                y_ang_rate_bytes = struct.pack('>HH', d[32], d[33])
                z_ang_rate_bytes = struct.pack('>HH', d[34], d[35])
                x_ang_rate = np.frombuffer(x_ang_rate_bytes, dtype=sgl_type)[0]
                y_ang_rate = np.frombuffer(y_ang_rate_bytes, dtype=sgl_type)[0]
                z_ang_rate = np.frombuffer(z_ang_rate_bytes, dtype=sgl_type)[0]

                att_t1 = d[36] * 2 ** 24
                att_t2 = (d[37] * 2 ** 8)
                att_t3 = (d[38] * 2 ** -8)
                att_t4 = (d[39] * 2 ** -24)
                att_t = att_t1 + att_t2 + att_t3 + att_t4

                output_dictionary.update({
                    "Q0 Attitude Quaternion": q0,
                    "Q1 Attitude Quaternion": q1,
                    "Q2 Attitude Quaternion": q2,
                    "Q3 Attitude Quaternion": q3,
                    "Omega-X Angular Rate": x_ang_rate,
                    "Omega-Y Angular Rate": y_ang_rate,
                    "Omega-Z Angular Rate": z_ang_rate,
                    "Attitude Data Timestamp": att_t
                })

                output_dict_list.append(output_dictionary)
    out_df = pd.DataFrame(output_dict_list)
    return out_df


def _ten_bit_unsigned_to_signed_int(ten_bit: int) -> int:
    """ From a ten-bit unsigned int, create a standard signed int.

        Ten_bit is an unprocessed 10-bit integer that was extracted from the packet.


        Returns: A signed integer in standard form
 """

    # The first bit is the sign, while the next nine bits are the number.
    sign = (-1) ** ((ten_bit >> 9) & 0x1)
    return sign * (ten_bit & 0x1ff)


def decode_bypass_data(data, num_quads):
    """Interpret user data of types A and B (also referred to as "Bypass" or "Decimation Only").

    Simply put, data is encoded as a string of ten-bit words.
.



    """
    num_words = math.ceil((10 / 16) * num_quads)  # number of words in 16 bits per channel
    num_bytes = 2 * num_words  # 8-bit byte count for each channel

    i_evens = np.zeros(num_quads)
    i_odds = np.zeros(num_quads)
    q_evens = np.zeros(num_quads)
    q_odds = np.zeros(num_quads)

    # It is difficult to extract 10-bit values from Python.
    #  Four 10-bit words are equal to 40 bits times five 8-bit bytes.
    # After reading each set of five standard 8-bit bytes, four 10-bit words will be extracted.
    # We'll need to monitor the indexing separately each time we check for the file's end.

    # Channel 1 - IE
    index_8bit = 0
    index_10bit = 0
    increase_count = 0
    left_shift_count = 2
    right_shift_count = 12
    while index_10bit < num_quads:
        if index_10bit < num_quads:
            s_code = (data[index_8bit + increase_count] << left_shift_count * 1 | data[
                index_8bit + increase_count + 1] >> int(right_shift_count / 2)) & 1023
            q_odds[index_10bit] = _ten_bit_unsigned_to_signed_int(s_code)
            increase_count += 1
            left_shift_count += 2
            right_shift_count -= 4
            index_10bit += 1
            index_8bit += 5
        if right_shift_count == 0:
            break

    # Channel 2 - IO
    index_8bit = num_bytes
    index_10bit = 0
    increase_count = 0
    left_shift_count = 2
    right_shift_count = 12
    while index_10bit < num_quads:
        if index_10bit < num_quads:
            s_code = (data[index_8bit + increase_count] << left_shift_count * 1 | data[
                index_8bit + increase_count + 1] >> int(right_shift_count / 2)) & 1023
            q_odds[index_10bit] = _ten_bit_unsigned_to_signed_int(s_code)
            increase_count += 1
            left_shift_count += 2
            right_shift_count -= 4
            index_10bit += 1
            index_8bit += 5
        if right_shift_count == 0:
            break

    # Channel 3 - QE
    index_8bit = 2 * num_bytes
    index_10bit = 0
    increase_count = 0
    left_shift_count = 2
    right_shift_count = 12
    while index_10bit < num_quads:
        if index_10bit < num_quads:
            s_code = (data[index_8bit + increase_count] << left_shift_count * 1 | data[
                index_8bit + increase_count + 1] >> int(right_shift_count / 2)) & 1023
            q_odds[index_10bit] = _ten_bit_unsigned_to_signed_int(s_code)
            increase_count += 1
            left_shift_count += 2
            right_shift_count -= 4
            index_10bit += 1
            index_8bit += 5
        if right_shift_count == 0:
            break

    # Channel 4 - QO
    index_8bit = 3 * num_bytes
    index_10bit = 0
    increase_count = 0
    left_shift_count = 2
    right_shift_count = 12
    while index_10bit < num_quads:
        if index_10bit < num_quads:
            s_code = (data[index_8bit + increase_count] << left_shift_count * 1 | data[
                index_8bit + increase_count + 1] >> int(right_shift_count / 2)) & 1023
            q_odds[index_10bit] = _ten_bit_unsigned_to_signed_int(s_code)
            increase_count += 1
            left_shift_count += 2
            right_shift_count -= 4
            index_10bit += 1
            index_8bit += 5
        if right_shift_count == 0:
            break

    return i_evens, i_odds, q_evens, q_odds


_TREE_BRC_ZERO = (0, (1, (2, 3)))
_TREE_BRC_ONE = (0, (1, (2, (3, 4))))
_TREE_BRC_TWO = (0, (1, (2, (3, (4, (5, 6))))))
_TREE_BRC_THREE = ((0, 1), (2, (3, (4, (5, (6, (7, (8, 9))))))))
_TREE_BRC_FOUR = ((0, (1, 2)), ((3, 4), ((5, 6), (7, (8, (9, ((10, 11), ((12, 13), (14, 15)))))))))

'''Some functionality of this code was adapted from
     this source https://github.com/jmfriedt/sentinel1_level0'''
class FDBAQDecoder:
    """extracts code samples from packets sent by Sentinel-1. """

    def __init__(self, data, num_quads):
        self._bit_counter = 0
        self._byte_counter = 0
        self._data = data
        self._num_quads = num_quads

        self._num_baq_blocks = math.ceil(num_quads / 128)
        self._brc = []
        self._thidx = []

        self._i_evens_scodes = []
        self._i_odds_scodes = []
        self._q_evens_scodes = []
        self._q_odds_scodes = []

        logging.debug(f"Created FDBAQ decoder. Numquads={num_quads} NumBAQblocks={self._num_baq_blocks}")

        # Channel 1 - IE
        values_processed_count = 0
        for block_index in range(self._num_baq_blocks):
            logging.debug(
                f"Starting IE block {block_index + 1} of {self._num_baq_blocks}, "
                f"processing {min(128, self._num_quads - values_processed_count)} vals")

            # The first three bits of every IE block contain the Bit Rate Code for each.
            brc = self._read_brc()
            self._brc.append(brc)

            # The type of Huffman encoding that we use is determined by the BRC.

            if self._brc[block_index] == 0:
                this_huffman_tree = _TREE_BRC_ZERO
            elif self._brc[block_index] == 1:
                this_huffman_tree = _TREE_BRC_ONE
            elif self._brc[block_index] == 2:
                this_huffman_tree = _TREE_BRC_TWO
            elif self._brc[block_index] == 3:
                this_huffman_tree = _TREE_BRC_THREE
            elif self._brc[block_index] == 4:
                this_huffman_tree = _TREE_BRC_FOUR
            else:
                logging.error(f"Unrecognized BAQ mode code {self._brc[block_index]}")

            # All baq blocks have 128 hcodes, with the exception of the last
            for i in range(min(128, self._num_quads - values_processed_count)):
                sign = self._next_bit()

                # Step backwards via our Huffman tree.
                # When our current node is an integer rather than a tuple, we know we've reached the end.

                current_node = this_huffman_tree
                while not isinstance(current_node, int):
                    current_node = current_node[self._next_bit()]
                    if current_node is None:
                        raise ValueError
                self._i_evens_scodes.append(SC(sign, current_node))
                values_processed_count = values_processed_count + 1

        # Channel 2 - IO
        # To the next 16-bit word boundary, move the counts.
        logging.debug(f"Finished block: bit_counter={self._bit_counter} byte_counter={self._byte_counter}")
        if not self._bit_counter == 0:
            self._bit_counter = 0
            self._byte_counter += 1
        self._byte_counter = math.ceil(self._byte_counter / 2) * 2
        logging.debug(f"Moved counters: bit_counter={self._bit_counter} byte_counter={self._byte_counter}")

        values_processed_count = 0
        for block_index in range(self._num_baq_blocks):
            logging.debug(
                f"Starting IO block {block_index + 1} of {self._num_baq_blocks}, "
                f"processing {min(128, self._num_quads - values_processed_count)} vals")

            # The type of Huffman encoding that we use is determined by the BRC.

            if self._brc[block_index] == 0:
                this_huffman_tree = _TREE_BRC_ZERO
            elif self._brc[block_index] == 1:
                this_huffman_tree = _TREE_BRC_ONE
            elif self._brc[block_index] == 2:
                this_huffman_tree = _TREE_BRC_TWO
            elif self._brc[block_index] == 3:
                this_huffman_tree = _TREE_BRC_THREE
            elif self._brc[block_index] == 4:
                this_huffman_tree = _TREE_BRC_FOUR
            else:
                logging.error(f"Unrecognized BAQ mode code {self._brc[block_index]}")


            for i in range(min(128, self._num_quads - values_processed_count)):
                sign = self._next_bit()


                current_node = this_huffman_tree
                while not isinstance(current_node, int):
                    current_node = current_node[self._next_bit()]
                    if current_node is None:
                        raise ValueError
                self._i_odds_scodes.append(SC(sign, current_node))
                values_processed_count = values_processed_count + 1

        # Channel 3 -QE

        logging.debug(f"Finished block: bit_counter={self._bit_counter} byte_counter={self._byte_counter}")
        if not self._bit_counter == 0:
            self._bit_counter = 0
            self._byte_counter += 1
        self._byte_counter = math.ceil(self._byte_counter / 2) * 2
        logging.debug(f"Moved counters: bit_counter={self._bit_counter} byte_counter={self._byte_counter}")

        values_processed_count = 0
        for block_index in range(self._num_baq_blocks):
            logging.debug(
                f"Starting QE block {block_index + 1} of {self._num_baq_blocks}, "
                f"processing {min(128, self._num_quads - values_processed_count)} vals")

            # The first eight bits of every IE block contain each THIDX Code.
            this_thidx = self._read_thidx()
            self._thidx.append(this_thidx)


            if self._brc[block_index] == 0:
                this_huffman_tree = _TREE_BRC_ZERO
            elif self._brc[block_index] == 1:
                this_huffman_tree = _TREE_BRC_ONE
            elif self._brc[block_index] == 2:
                this_huffman_tree = _TREE_BRC_TWO
            elif self._brc[block_index] == 3:
                this_huffman_tree = _TREE_BRC_THREE
            elif self._brc[block_index] == 4:
                this_huffman_tree = _TREE_BRC_FOUR
            else:
                logging.error(f"Unrecognized BAQ mode code {self._brc[block_index]}")


            for i in range(min(128, self._num_quads - values_processed_count)):
                sign = self._next_bit()


                current_node = this_huffman_tree
                while not isinstance(current_node, int):
                    current_node = current_node[self._next_bit()]
                    if current_node is None:
                        raise ValueError
                self._q_evens_scodes.append(SC(sign, current_node))
                values_processed_count = values_processed_count + 1

        # Channel 4 - QO

        logging.debug(f"Finished block: bit_counter={self._bit_counter} byte_counter={self._byte_counter}")
        if not self._bit_counter == 0:
            self._bit_counter = 0
            self._byte_counter += 1
        self._byte_counter = math.ceil(self._byte_counter / 2) * 2
        logging.debug(f"Moved counters: bit_counter={self._bit_counter} byte_counter={self._byte_counter}")

        values_processed_count = 0
        for block_index in range(self._num_baq_blocks):
            logging.debug(
                f"Starting QO block {block_index + 1} of {self._num_baq_blocks}, "
                f"processing {min(128, self._num_quads - values_processed_count)} vals")


            if self._brc[block_index] == 0:
                this_huffman_tree = _TREE_BRC_ZERO
            elif self._brc[block_index] == 1:
                this_huffman_tree = _TREE_BRC_ONE
            elif self._brc[block_index] == 2:
                this_huffman_tree = _TREE_BRC_TWO
            elif self._brc[block_index] == 3:
                this_huffman_tree = _TREE_BRC_THREE
            elif self._brc[block_index] == 4:
                this_huffman_tree = _TREE_BRC_FOUR
            else:
                logging.error(f"Unrecognized BAQ mode code {self._brc[block_index]}")


            for i in range(min(128, self._num_quads - values_processed_count)):
                sign = self._next_bit()


                current_node = this_huffman_tree
                while not isinstance(current_node, int):
                    current_node = current_node[self._next_bit()]
                    if current_node is None:
                        raise ValueError
                self._q_odds_scodes.append(SC(sign, current_node))
                values_processed_count = values_processed_count + 1

    @property
    def get_brcs(self):
        """Obtain the Bit Rate Codes (BRCs) extracted list.."""
        return self._brc

    @property
    def get_thidxs(self):
        """Obtain the Threshold Index code (THIDX) extraction list.."""
        return self._thidx

    @property
    def get_s_ie(self):
        """Obtain the I channel even-indexed data."""
        return self._i_evens_scodes

    @property
    def get_s_io(self):
        """Obtain the I channel odd-indexed data."""
        return self._i_odds_scodes

    @property
    def get_s_qe(self):
        """Obtain the Q channel even-indexed data."""
        return self._q_evens_scodes

    @property
    def get_s_qo(self):
        """Obtain the Q channel odd-indexed data."""
        return self._q_odds_scodes

    def _next_bit(self):
        bit = (self._data[self._byte_counter] >> (7 - self._bit_counter)) & 0x01
        self._bit_counter = (self._bit_counter + 1) % 8
        if self._bit_counter == 0:
            self._byte_counter += 1
        return bit

    def _read_thidx(self):
        residual = 0
        for i in range(8):
            residual = residual << 1
            residual += self._next_bit()
        return residual

    def _read_brc(self):
        residual = 0
        for i in range(3):
            residual = residual << 1
            residual += self._next_bit()
        return residual


def reconstruct_values(data, brcs, thidxs, values_to_process):
    if not len(brcs) == len(thidxs):
        print("Mismatched lengths of BRC block parameters")

    num_brc_blocks = len(brcs)
    out_vals = np.zeros(values_to_process)
    n = 0

    brc_mappings = {
        0: (b0, nrl_b0),
        1: (b1, nrl_b1),
        2: (b2, nrl_b2),
        3: (b3, nrl_b3),
        4: (b4, nrl_b4)
    }

    for block_index in range(num_brc_blocks):
        brc = int(brcs[block_index])
        thidx = int(thidxs[block_index])

        if brc not in brc_mappings:
            print("Invalid BRC value")
            continue

        code_range, nrl_list = brc_mappings[brc]
        sf_value = sf[thidx]

        for i in range(min(128, values_to_process - n)):
            s_code = data[n]
            get_mcode = s_code['get_mcode']
            get_sign = s_code['get_sign']

            if thidx <= len(code_range) - 1:
                if get_mcode < len(code_range):
                    out_vals[n] = (-1) ** get_sign * get_mcode
                elif get_mcode == len(code_range):
                    out_vals[n] = (-1) ** get_sign * code_range[thidx]
                else:
                    print('error')
            else:
                if get_mcode < len(nrl_list):
                    out_vals[n] = (-1) ** get_sign * get_mcode
                elif get_mcode == len(nrl_list):
                    out_vals[n] = (-1) ** get_sign * nrl_list[thidx]
                else:
                    print('error')

            n += 1

    return out_vals


def SC(sign, mcode):  # Sample Code
    SampleCode = {}
    SampleCode['get_sign'] = sign
    SampleCode['get_mcode'] = mcode
    return SampleCode


def create_dataframe(
        file_address: str,
        file_name: str,
        dataframe_name: str,
        save_df_path: str

) -> tuple[DataFrame, str, str]:
    start_time = time.time()
    logging.info('################## Creating Dataframe ###################')
    output_packet_ = {}
    row_list = []
    F_REF = 37.53472224 * 1e6
    with open(file_address + f"{file_name}", "rb") as f:
        while True:
            header = f.read(6)
            if not header:
                break
            if len(header) != 6:
                print('Invalid')
            else:
                tmp16 = int.from_bytes(header[:2],
                                       'big')
                packet_version_number = tmp16 >> 13
                packet_type = (tmp16 >> 12) & 0x01
                secondary_header_flag = (tmp16 >> 11) & 0x01
                process_id = (tmp16 >> 4) & 0x7f
                packet_category = tmp16 & 0xf
                tmp16 = int.from_bytes(header[2:4], 'big')
                sequence_flags = tmp16 >> 14
                packet_sequence_count = tmp16 & 0x3f
                tmp16 = int.from_bytes(header[4:], 'big')
                packet_data_length = tmp16 + 1

                output_packet_['packet_version_number'] = packet_version_number
                output_packet_['packet_type'] = packet_type
                output_packet_['secondary_header_flag'] = secondary_header_flag
                output_packet_['process_id'] = process_id
                output_packet_['packet_category'] = packet_category
                output_packet_['sequence_flags'] = sequence_flags
                output_packet_['packet_sequence_count'] = packet_sequence_count
                output_packet_['packet_data_length'] = packet_data_length

                pk_data_length = output_packet_['packet_data_length']
                data_ = f.read(pk_data_length)
                if len(data_[:62]) != 62:
                    print('invalid')

                coarse_time = int.from_bytes(data_[:4], 'big')
                fine_time = (int.from_bytes(data_[4:6], 'big') + 0.5) * (2 ** (-16))
                sync = int.from_bytes(data_[6:10], 'big')

                data_take_id = int.from_bytes(data_[10:14], 'big')

                ecc_number = data_[14]


                test_mode = (data_[15] >> 4) & 0x07
                rx_channel_id = data_[15] & 0x0f

                instrument_config_id = int.from_bytes(data_[16:20], 'big')
                subcom_data_word_ind = data_[20]
                subcom_data_word = int.from_bytes(data_[21:23], 'big')
                space_packet_count = int.from_bytes(data_[23:27], 'big')
                pri_count = int.from_bytes(data_[27:31], 'big')
                error_flag = data_[31] >> 7

                baq_mode = data_[31] & 0x1f

                baq_block_length = data_[32]



                range_decimation = data_[34]

                rx_gain = data_[35] * -0.5

                tmp16 = int.from_bytes(data_[36:38], 'big')
                txprr_sign = ((-1) ** (1 - (tmp16 >> 15)))
                txprr = txprr_sign * (tmp16 & 0x7fff) * (F_REF ** 2) / (2 ** 21)

                tmp16 = int.from_bytes(data_[38:40], 'big')
                txpsf_additive = (txprr / (4 * F_REF))
                txpsf_sign = ((-1) ** (1 - (tmp16 >> 15)))
                txpsf = txpsf_additive + txpsf_sign * (tmp16 & 0x7fff) * F_REF / (2 ** 14)

                tmp24 = int.from_bytes(data_[40:43], 'big')

                tx_pulse_length = tmp24 / F_REF


                rank = data_[43] & 0x1f  # Byte 43 bits 3-7

                tmp24 = int.from_bytes(data_[44:47], 'big')
                pri = tmp24 / F_REF

                tmp24 = int.from_bytes(data_[47:50], 'big')
                sampling_window_start_time = tmp24 / F_REF

                tmp24 = int.from_bytes(data_[50:53], 'big')
                sampling_window_length = tmp24 / F_REF

                sas_ssbflag = data_[53] >> 7
                polarisation = (data_[53] >> 4) & 0x07
                temperature_comp = (data_[53] >> 2) & 0x03


                calibration_mode = data_[56] >> 6

                tx_pulse_number = data_[56] & 0x1f
                signal_type = data_[57] >> 4

                swap_flag = data_[57] & 0x01

                swath_number = data_[58]

                number_of_quads = int.from_bytes(data_[59:61], 'big')

                output_packet_['coarse_time'] = coarse_time
                output_packet_['fine_time'] = fine_time

                output_packet_['sync'] = sync
                output_packet_['data_take_id'] = data_take_id
                output_packet_['ecc_number'] = ecc_number
                output_packet_['test_mode'] = test_mode
                output_packet_['rx_channel_id'] = rx_channel_id
                output_packet_['instrument_config_id'] = instrument_config_id
                output_packet_['subcom_data_word_ind'] = subcom_data_word_ind
                output_packet_['subcom_data_word'] = subcom_data_word
                output_packet_['space_packet_count'] = space_packet_count
                output_packet_['error_flag'] = error_flag
                output_packet_['baq_mode'] = baq_mode
                output_packet_['baq_block_length'] = baq_block_length
                output_packet_['range_decimation'] = range_decimation
                output_packet_['rx_gain'] = rx_gain
                output_packet_['txprr_sign'] = txprr_sign
                output_packet_['txprr'] = txprr
                output_packet_['txpsf_additive'] = txpsf_additive
                output_packet_['txpsf_sign'] = txpsf_sign
                output_packet_['txpsf'] = txpsf
                output_packet_['tmp24'] = tmp24
                output_packet_['tx_pulse_length'] = tx_pulse_length
                output_packet_['rank'] = rank
                output_packet_['pri'] = pri
                output_packet_['sampling_window_start_time'] = sampling_window_start_time
                output_packet_['sampling_window_length'] = sampling_window_length
                output_packet_['sas_ssbflag'] = sas_ssbflag
                output_packet_['polarisation'] = polarisation
                output_packet_['temperature_comp'] = temperature_comp
                output_packet_['calibration_mode'] = calibration_mode
                output_packet_['tx_pulse_number'] = tx_pulse_number
                output_packet_['signal_type'] = signal_type
                output_packet_['swap_flag'] = swap_flag
                output_packet_['swath_number'] = swath_number
                output_packet_['number_of_quads'] = number_of_quads
                row_list.append(output_packet_.copy())

    pd.DataFrame(row_list).to_csv(os.path.join(save_df_path, f'{dataframe_name}.csv'), index=False)
    end_time = time.time()
    logging.info(f"Time taken: {end_time - start_time:.2f} seconds")
    return pd.DataFrame(row_list), dataframe_name, save_df_path


def decoding_packets_PH_SH(
        meta_dataframe: pd.DataFrame,
        file_address: str,
        file_name: str
) -> np.ndarray:

    packet_count = 0
    start_time = time.time()
    logging.info('################## Decoding_packets_PH_SH ###################')
    nq_ = meta_dataframe['number_of_quads'].unique()[0]

    number_of_packets = len(meta_dataframe)
    out_output_bytes = np.zeros([number_of_packets, nq_ * 2], dtype=(complex))

    with open(file_address + f"{file_name}", "rb") as f:
        while True:
            output_packet = {}
            header = f.read(6)
            if not header:
                break
            if len(header) != 6:
                print('Invalid')
            else:
                tmp16 = int.from_bytes(header[:2],
                                       'big')
                packet_version_number = tmp16 >> 13
                packet_type = (tmp16 >> 12) & 0x01
                secondary_header_flag = (tmp16 >> 11) & 0x01
                process_id = (tmp16 >> 4) & 0x7f
                packet_category = tmp16 & 0xf
                tmp16 = int.from_bytes(header[2:4], 'big')
                sequence_flags = tmp16 >> 14
                packet_sequence_count = tmp16 & 0x3f
                tmp16 = int.from_bytes(header[4:], 'big')
                packet_data_length = tmp16 + 1

                output_packet['packet_version_number'] = packet_version_number
                output_packet['packet_type'] = packet_type
                output_packet['secondary_header_flag'] = secondary_header_flag
                output_packet['process_id'] = process_id
                output_packet['packet_category'] = packet_category
                output_packet['sequence_flags'] = sequence_flags
                output_packet['packet_sequence_count'] = packet_sequence_count
                output_packet['packet_data_length'] = packet_data_length

                pk_data_length = output_packet['packet_data_length']
                data_ = f.read(pk_data_length)

                if len(data_[:62]) != 62:
                    print('invalid')

                coarse_time = int.from_bytes(data_[:4], 'big')
                fine_time = (int.from_bytes(data_[4:6], 'big') + 0.5) * (2 ** (-16))
                sync = int.from_bytes(data_[6:10], 'big')

                data_take_id = int.from_bytes(data_[10:14], 'big')

                ecc_number = data_[14]


                test_mode = (data_[15] >> 4) & 0x07
                rx_channel_id = data_[15] & 0x0f

                instrument_config_id = int.from_bytes(data_[16:20], 'big')
                subcom_data_word_ind = data_[20]
                subcom_data_word = int.from_bytes(data_[21:23], 'big')
                space_packet_count = int.from_bytes(data_[23:27], 'big')
                pri_count = int.from_bytes(data_[27:31], 'big')
                error_flag = data_[31] >> 7

                baq_mode = data_[31] & 0x1f

                baq_block_length = data_[32]


                range_decimation = data_[34]

                rx_gain = data_[35] * -0.5

                tmp16 = int.from_bytes(data_[36:38], 'big')
                txprr_sign = ((-1) ** (1 - (tmp16 >> 15)))
                txprr = txprr_sign * (tmp16 & 0x7fff) * (F_REF ** 2) / (2 ** 21)

                tmp16 = int.from_bytes(data_[38:40], 'big')
                txpsf_additive = (txprr / (4 * F_REF))
                txpsf_sign = ((-1) ** (1 - (tmp16 >> 15)))
                txpsf = txpsf_additive + txpsf_sign * (tmp16 & 0x7fff) * F_REF / (2 ** 14)

                tmp24 = int.from_bytes(data_[40:43], 'big')

                tx_pulse_length = tmp24 / F_REF


                rank = data_[43] & 0x1f

                tmp24 = int.from_bytes(data_[44:47], 'big')
                pri = tmp24 / F_REF

                tmp24 = int.from_bytes(data_[47:50], 'big')
                sampling_window_start_time = tmp24 / F_REF

                tmp24 = int.from_bytes(data_[50:53], 'big')
                sampling_window_length = tmp24 / F_REF

                sas_ssbflag = data_[53] >> 7
                polarisation = (data_[53] >> 4) & 0x07
                temperature_comp = (data_[53] >> 2) & 0x03


                calibration_mode = data_[56] >> 6

                tx_pulse_number = data_[56] & 0x1f
                signal_type = data_[57] >> 4

                swap_flag = data_[57] & 0x01

                swath_number = data_[58]

                number_of_quads = int.from_bytes(data_[59:61], 'big')

                output_packet['coarse_time'] = coarse_time
                output_packet['fine_time'] = fine_time

                output_packet['sync'] = sync
                output_packet['data_take_id'] = data_take_id
                output_packet['ecc_number'] = ecc_number
                output_packet['test_mode'] = test_mode
                output_packet['rx_channel_id'] = rx_channel_id
                output_packet['instrument_config_id'] = instrument_config_id
                output_packet['subcom_data_word_ind'] = subcom_data_word_ind
                output_packet['subcom_data_word'] = subcom_data_word
                output_packet['space_packet_count'] = space_packet_count
                output_packet['error_flag'] = error_flag
                output_packet['baq_mode'] = baq_mode
                output_packet['baq_block_length'] = baq_block_length
                output_packet['range_decimation'] = range_decimation
                output_packet['rx_gain'] = rx_gain
                output_packet['txprr_sign'] = txprr_sign
                output_packet['txprr'] = txprr
                output_packet['txpsf_additive'] = txpsf_additive
                output_packet['txpsf_sign'] = txpsf_sign
                output_packet['txpsf'] = txpsf
                output_packet['tmp24'] = tmp24
                output_packet['tx_pulse_length'] = tx_pulse_length
                output_packet['rank'] = rank
                output_packet['pri'] = pri
                output_packet['sampling_window_start_time'] = sampling_window_start_time
                output_packet['sampling_window_length'] = sampling_window_length
                output_packet['sas_ssbflag'] = sas_ssbflag
                output_packet['polarisation'] = polarisation
                output_packet['temperature_comp'] = temperature_comp
                output_packet['calibration_mode'] = calibration_mode
                output_packet['tx_pulse_number'] = tx_pulse_number
                output_packet['signal_type'] = signal_type
                output_packet['swap_flag'] = swap_flag
                output_packet['swath_number'] = swath_number
                output_packet['number_of_quads'] = number_of_quads
                output_bytes = data_[62:]
                if output_packet['space_packet_count'] in meta_dataframe['space_packet_count'].values:
                    baqmod = output_packet['baq_mode']
                    nq = output_packet['number_of_quads']
                    if baqmod == 12 or baqmod == 13 or baqmod == 14:

                        scode_extractor = FDBAQDecoder(output_bytes, nq)
                        brcs = scode_extractor.get_brcs
                        thidxs = scode_extractor.get_thidxs

                        IE = reconstruct_values(
                            scode_extractor.get_s_ie, brcs, thidxs, nq
                        )
                        IO = reconstruct_values(
                            scode_extractor.get_s_io, brcs, thidxs, nq
                        )
                        QE = reconstruct_values(
                            scode_extractor.get_s_qe, brcs, thidxs, nq
                        )
                        QO = reconstruct_values(
                            scode_extractor.get_s_qo, brcs, thidxs, nq
                        )

                        decoded_data = []
                        for i in range(len(IE)):
                            decoded_data.append(complex(IE[i], QE[i]))
                            decoded_data.append(complex(IO[i], QO[i]))
                    elif baqmod == 0:

                        IE, IO, QE, QO = decode_bypass_data(output_bytes, nq)

                        decoded_data = []
                        for i in range(len(IE)):
                            decoded_data.append(complex(IE[i], QE[i]))
                            decoded_data.append(complex(IO[i], QO[i]))

                    else:

                        continue

                    if decoded_data:
                        if len(decoded_data) > nq_ * 2:
                            out_output_bytes[packet_count, :] = decoded_data[:nq_ * 2]
                            packet_count = packet_count + 1
                        else:
                            out_output_bytes[packet_count, :len(decoded_data)] = decoded_data
                            packet_count = packet_count + 1
                    else:
                        out_output_bytes[packet_count, :] = 0
                        packet_count = packet_count + 1

                else:
                    out_output_bytes[packet_count, :] = 0
                    packet_count = packet_count + 1
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 3600
    elapsed_time = int((elapsed_time % 1) * 100)
    elapsed_hour = elapsed_time / 60
    hours = int(elapsed_hour)
    minutes = int((elapsed_hour % 1) * 60)
    logging.info(f"Time taken: {hours} hour, {minutes} minutes")
    return out_output_bytes


def plot_azimuth_data(
        file_address: str,
        file_name: str,
        save_fig_path: str
):
    az_compressed_data = np.load(os.path.join(file_address, f'{file_name}.npy'))
    plt.figure(figsize=(16, 100))
    plt.title("Sentinel-1 Processed SAR Image")
    plt.imshow(abs(az_compressed_data[:, 5500:10000]), origin='lower', vmin=100, vmax=500,
               cmap='gray')  # ,aspect='auto')
    plt.xlabel("Down Range (samples)")
    plt.ylabel("Cross Range (samples)")
    plt.savefig(os.path.join(save_fig_path, f'compressed_{file_name}.png'), bbox_inches='tight')


def save_az_compressed_data(
        array: np.ndarray,
        filename: str,
        save_path_address: str
):
    np.save(os.path.join(save_path_address, f'az_compressed_data_{filename}.npy'), array)
# some functionalities of code are inspired from these github pages:
#https://github.com/Rich-Hall/sentinel1decoder
#https://nbviewer.org/github/Rich-Hall/sentinel1Level0DecodingDemo/blob/main/sentinel1Level0DecodingDemo.ipynb
