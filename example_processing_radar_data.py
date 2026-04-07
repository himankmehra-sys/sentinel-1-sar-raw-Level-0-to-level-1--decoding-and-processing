from helper_functions import *
from DataProcessingRadarData import RadarDataProcessing
from tqdm import tqdm
import joblib

'''Read the dictionery saved from the read_sentinel_data to access the saved filepath and names used.'''

stored_address_and_filename = joblib.load('D:\\temp_path_rd\\stored_address_and_filename.joblib')

chunk_files = os.listdir(stored_address_and_filename['save_chunks_path_address'])

for file in tqdm(chunk_files):
    file_name = file.split('.')[0]
    if len(np.load(f'{stored_address_and_filename['save_chunks_path_address']}\\{file}')) != 0:

        #read chunks of data saved
        radar_data=read_saved_radar_data(saved_radar_data_path=stored_address_and_filename['save_chunks_path_address'],
                                         filename=file_name)

        tiles_raw_iq_array = tiles_of_data(radar_data=radar_data,start_limit = 0,end_limit=len(radar_data))

        #read saved meta data dataframe
        meta_df = read_dataframe(file_address=stored_address_and_filename['saved_path'],
                                 file_name=stored_address_and_filename['dataframe_name'])

        ephemeris = read_subcommed_data(meta_df)


        RDP=RadarDataProcessing(meta_dataframe=meta_df,
                            ephemeris=ephemeris,
                            tile_array=tiles_raw_iq_array)

        D, slant_range, az_freq_vals=RDP.squint_angle_calc()

        freq_domain_data = RDP.conversion_time_frq_domain()

        freq_domain_data=RDP.range_filter_from_replica_pulse(freq_domain_data=freq_domain_data)

        rcmc_filt = RDP.rcmc_filter(D=D,
                                    slant_range=slant_range,
                                    freq_domain_data=freq_domain_data)

        range_doppler_data = RDP.doppler_correction(freq_domain_data=freq_domain_data)

        az_compressed_data = RDP.azimuth_compression(D=D,
                                                range_doppler_data=range_doppler_data,
                                                az_freq_vals=az_freq_vals,
                                                slant_range=slant_range)

        save_az_compressed_data(array=az_compressed_data,
                                filename=file_name,
                                save_path_address=stored_address_and_filename['saved_path'])

        plot_azimuth_data(file_address=stored_address_and_filename['saved_path'],
                          file_name=f'az_compressed_data_{file_name}',
                          save_fig_path=stored_address_and_filename['saved_path'])


exit('---Success---')



