import matplotlib.pyplot as plt
import pandas as pd
import yaml
import os


with open('graphing_configuration.yaml', 'r') as f:
    configs = yaml.safe_load(f)

###########################################
# make one big dataframe
###########################################
# Specify the folder containing the CSV files
# folder_path = configs['folder_path']

# Initialize an empty list to hold the dataframes
# dataframes = []

# Loop over all files in the folder
# for filename in os.listdir(folder_path):
#     if filename.endswith('.csv'):  # Check if the file is a CSV
#         file_path = os.path.join(folder_path, filename)  # Get the full path to the file
#         df = pd.read_csv(file_path)  # Read the CSV file into a dataframe
#         if configs['keep_only'] != 0:
#             df = df.head(configs['keep_only'])  # Keep only the first 200 samples
#         dataframes.append(df)  # Append the dataframe to the list

# Concatenate all dataframes in the list into a single large dataframe
# combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined dataframe to a new CSV file
# combined_df.to_csv(configs['output_file_name'], index=False)

# print("Combined DataFrame saved to 'combined_output.csv'")


########################################################
# Orientation Histrogram
######################################################
# master_file = pd.read_csv(configs['monte_carlo_results_file_path'], sep=',', header=0, names=configs['monte_carlo_results_column_names'])
# figure = plt.figure()
# plt.hist(master_file['rmse_q'], bins=configs['bins'])
# plt.xlabel('Orientation RMSE (degrees)')
# plt.ylabel('Number of Runs')
# plt.show()


#######################################################
# Orientation selection summary
######################################################


#######################################################
# Good bad error orientation cluster plots
#######################################################

def extract_number(s):
    num_str = ''.join(filter(str.isdigit, s))
    return int(num_str) if num_str else None

# open rmse file
master_file = pd.read_csv(configs['monte_carlo_results_file_path'], sep=',', header=0, names=configs['monte_carlo_results_column_names'])

# get good ones
good_files = master_file[master_file['rmse_q'] <= configs['success_thresh_angle']]

# get bad ones
bad_files = master_file[master_file['rmse_q'] > configs['success_thresh_angle']]

# clip to get file number and convert to int
good_files['pickle_file'] = good_files['pickle_file'].apply(extract_number)
bad_files['pickle_file'] = bad_files['pickle_file'].apply(extract_number)

# open initial conditions file
ini_cond_file = pd.read_csv(configs['initial_conditions_file_path'], sep=',', header=0, names=configs['initial_conditions_column_names'])

# get good and bad population
good_ini_cond = ini_cond_file[ini_cond_file['file index'].isin(good_files['pickle_file'])]
bad_ini_cond = ini_cond_file[ini_cond_file['file index'].isin(bad_files['pickle_file'])]

print(good_ini_cond)

# plotting loop
for idx, column in enumerate(configs['initial_conditions_column_names']):
    if ('nframes' in column) or ('file index' in column):
        pass
    else:
        figure = plt.figure()
        plt.scatter(good_ini_cond[column], good_files['rmse_q'], s=1, color='blue')
        plt.scatter(bad_ini_cond[column], bad_files['rmse_q'], s=1, color='red')
        plt.xlabel(column)
        plt.ylabel('rmse q (degrees)')


plt.show()



##########################################
# feature scatter plots with dataframe --- not integrated with config file yet
##########################################
# master_file = pd.read_csv('combined_output.csv', sep=',', header=0, names=configs['master_file_columns'])
# print(len(master_file['file_name']))
# for idx, column in enumerate(configs['master_file_columns']):
#
#     if ('ransac' in column) or ('pca' in column) or ('prediction' in column) or ('file_name' in column) or ('perfect_metric_choice' in column):
#         pass
#     else:
#         errors = ['ransac_error', 'pca_error', 'prediction_error']
#         for jdx, error in enumerate(errors):
#             figure = plt.figure()
#             plt.scatter(master_file[column], master_file[error], s=1)
#             plt.xlabel(column)
#             plt.ylabel(error)
#             plt.ylim([0, 200])
#             plt.savefig('graphs/' + column + error + '.png', format='png')
#             plt.close()
#
# ransac_inliers = master_file[master_file['perfect_metric_choice'] == 'ransac']
# print("Ransac Inliers:" + str(len(ransac_inliers)))
# ransac_outliers = master_file[master_file['perfect_metric_choice'] != 'ransac']
# print("Ransac Outliers:" +str(len(ransac_outliers)))
# pca_inliers = ransac_outliers[ransac_outliers['perfect_metric_choice'] == 'pca']
# print("PCA_inliers: " + str(len(pca_inliers)))
# prediction_inliers = ransac_outliers[ransac_outliers['perfect_metric_choice'] == 'prediction']
# print("Prediction: " + str(len(prediction_inliers)))
#
# set_names = ['ransac_inliers', 'ransac_outliers', 'pca_inliers', 'prediction_inliers']
# all_sets = [ransac_inliers, ransac_outliers, pca_inliers, prediction_inliers]
# for kdx, set in enumerate(all_sets):
#     for idx, column in enumerate(configs['master_file_columns']):
#
#         if ('ransac' in column) or ('pca' in column) or ('prediction' in column) or ('file_name' in column) or (
#                 'perfect_metric_choice' in column):
#             pass
#         else:
#             errors = ['ransac_error', 'pca_error', 'prediction_error']
#             for jdx, error in enumerate(errors):
#                 figure = plt.figure()
#                 plt.scatter(set[column], set[error], s=1)
#                 plt.xlabel(column)
#                 plt.ylabel(error)
#                 plt.ylim([0,200])
#                 plt.title(set_names[kdx])
#                 plt.savefig('graphs/' + set_names[kdx] + column + error + '.png', format='png')
#                 plt.close()
#
#
#
#