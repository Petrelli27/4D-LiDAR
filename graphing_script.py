import matplotlib.pyplot as plt
import pandas as pd
import yaml
import os


with open('configuration.yaml', 'r') as f:
    configs = yaml.safe_load(f)

###########################################
# make one big dataframe
###########################################
# Specify the folder containing the CSV files
# folder_path = 'full_results'

# Initialize an empty list to hold the dataframes
# dataframes = []

# Loop over all files in the folder
# for filename in os.listdir(folder_path):
#     if filename.endswith('.csv'):  # Check if the file is a CSV
#         file_path = os.path.join(folder_path, filename)  # Get the full path to the file
#         df = pd.read_csv(file_path)  # Read the CSV file into a dataframe
#         dataframes.append(df)  # Append the dataframe to the list

# Concatenate all dataframes in the list into a single large dataframe
# combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined dataframe to a new CSV file
# combined_df.to_csv('combined_output.csv', index=False)

# print("Combined DataFrame saved to 'combined_output.csv'")


##########################################
# plot with dataframe
##########################################
master_file = pd.read_csv('combined_output.csv', sep=',', header=0, names=configs['master_file_columns'])
print(len(master_file['file_name']))
for idx, column in enumerate(configs['master_file_columns']):

    if ('ransac' in column) or ('pca' in column) or ('prediction' in column) or ('file_name' in column) or ('perfect_metric_choice' in column):
        pass
    else:
        errors = ['ransac_error', 'pca_error', 'prediction_error']
        for jdx, error in enumerate(errors):
            figure = plt.figure()
            plt.scatter(master_file[column], master_file[error], s=1)
            plt.xlabel(column)
            plt.ylabel(error)
            plt.ylim([0, 200])
            plt.savefig('graphs/' + column + error + '.png', format='png')
            plt.close()

ransac_inliers = master_file[master_file['perfect_metric_choice'] == 'ransac']
print("Ransac Inliers:" + str(len(ransac_inliers)))
ransac_outliers = master_file[master_file['perfect_metric_choice'] != 'ransac']
print("Ransac Outliers:" +str(len(ransac_outliers)))
pca_inliers = ransac_outliers[ransac_outliers['perfect_metric_choice'] == 'pca']
print("PCA_inliers: " + str(len(pca_inliers)))
prediction_inliers = ransac_outliers[ransac_outliers['perfect_metric_choice'] == 'prediction']
print("Prediction: " + str(len(prediction_inliers)))

set_names = ['ransac_inliers', 'ransac_outliers', 'pca_inliers', 'prediction_inliers']
all_sets = [ransac_inliers, ransac_outliers, pca_inliers, prediction_inliers]
for kdx, set in enumerate(all_sets):
    for idx, column in enumerate(configs['master_file_columns']):

        if ('ransac' in column) or ('pca' in column) or ('prediction' in column) or ('file_name' in column) or (
                'perfect_metric_choice' in column):
            pass
        else:
            errors = ['ransac_error', 'pca_error', 'prediction_error']
            for jdx, error in enumerate(errors):
                figure = plt.figure()
                plt.scatter(set[column], set[error], s=1)
                plt.xlabel(column)
                plt.ylabel(error)
                plt.ylim([0,200])
                plt.title(set_names[kdx])
                plt.savefig('graphs/' + set_names[kdx] + column + error + '.png', format='png')
                plt.close()



