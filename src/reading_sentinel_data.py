from helper_functions import *

''' ------------------------ Refer to example_read_sentinel_data.py to run ------------------------ '''
file_address = 'Enter file address here:'
filename = 'Enter file here example: .dat file'
stored_address_and_filename = {}
output_dataframe, dataframe_name, save_df_path = create_dataframe(file_address=file_address,
                                                                  file_name=filename,
                                                                  dataframe_name='Enter dataframe name to be saved',
                                                                  save_df_path='Path to save dataframe after processing')

radar_data = decoding_packets_PH_SH(meta_dataframe=output_dataframe,
                                    file_address=file_address,
                                    file_name=filename)

save_radar_data_file_address = save_radar_data(radar_data=radar_data,
                                               save_radar_data_address=save_df_path)

save_chunks_path_address = save_chunks_of_data(radar_data=radar_data,
                                               save_chunks_path=save_df_path)

stored_address_and_filename['dataframe_name'] = dataframe_name
stored_address_and_filename['saved_path'] = save_df_path
stored_address_and_filename['save_radar_data_file_address'] = save_radar_data_file_address
stored_address_and_filename['save_chunks_path_address'] = save_chunks_path_address

# We stored the data in the same folder, where we saved the meta dataframe
save_dictionary(stored_address_and_filename,
                fileaddress=save_df_path,
                filename='stored_address_and_filename')

exit('---Success---')
