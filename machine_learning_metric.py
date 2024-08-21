import pandas as pd
import numpy as np
import sklearn.metrics
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn import preprocessing
import matplotlib.gridspec as gridspec
from sklearn import svm
from sklearn.pipeline import make_pipeline
import multiprocessing
import sklearn as sk
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
import numpy as np
import yaml
import joblib


@staticmethod
def preprocessing_main(X, y):
    random_state = 42
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, stratify=y,
                                                        random_state=random_state)

    # classes = set(train_y)
    # indices = [np.where(train_y == class_i) for i, class_i in enumerate(classes)]
    # num_in_class = [len(indices_i[0]) for j, indices_i in enumerate(indices)]
    # data_Xy = np.hstack((train_X, np.array([train_y]).T))
    # data_classes = []
    # for j, value in enumerate(classes):
    #     data_class_i = data_Xy[indices[j], :].reshape((num_in_class[j], len(data_Xy[0])))
    #     data_classes.append(data_class_i)
    # augmented_data_Xy = data_Xy.copy()
    #
    # oversampling
    # max_samples = max(num_in_class)
    # for i, val in enumerate(num_in_class):
    #     if val == max_samples:  # you do not need to augment this class because it is already the largest
    #         pass
    #     else:
    #         ratio = int(max_samples / val)
    #         for j in range(ratio - 1):
    #             augmented_data_Xy = np.vstack((augmented_data_Xy, data_classes[i]))
    #
    # undersampling
    # min_samples = min(num_in_class)
    # undersampled_data_Xy = data_Xy[0, :]
    # for i, val in enumerate(num_in_class):
    #     undersampled_data_Xy = np.vstack((undersampled_data_Xy, data_classes[i][:min_samples, :]))
    #
    # np.random.shuffle(augmented_data_Xy)
    # np.random.shuffle(undersampled_data_Xy)

    # Under and over sampled
    # over_train_X = augmented_data_Xy[:, :-1]
    # over_train_y = augmented_data_Xy[:, -1].ravel()
    # under_train_X = undersampled_data_Xy[:, :-1]
    # under_train_y = undersampled_data_Xy[:, -1].ravel()
    #
    # indices_new = [np.where(over_train_y == class_i) for i, class_i in enumerate(classes)]
    # num_in_class_new = [len(indices_i[0]) for j, indices_i in enumerate(indices_new)]
    #
    # indices_new_under = [np.where(under_train_y == class_i) for i, class_i in enumerate(classes)]
    # num_in_class_new_under = [len(indices_i[0]) for j, indices_i in enumerate(indices_new_under)]
    #
    # print('Orignal class Distribution: ' + str(num_in_class))
    # print('Oversampled class distribution: ' + str(num_in_class_new))
    # print('Undersampled class distribution: ' + str(num_in_class_new_under) + '\n')
    #
    # Quantile transform
    # quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
    # quantile_train_X = quantile_transformer.fit_transform(train_X)
    # quantile_test_X = quantile_transformer.fit_transform(test_X)
    # quantile_train_oversampled_X = quantile_transformer.fit_transform(over_train_X)

    # Standard scalar
    # scalar_train_X = preprocessing.StandardScaler().fit(train_X)
    # standardscalar_train_X = scalar_train_X.transform(train_X)
    # scalar_test_X = preprocessing.StandardScaler().fit(test_X)
    # standardscalar_test_X = scalar_test_X.transform(test_X)

    # Normalized
    normalized_train_X = preprocessing.normalize(train_X, norm='l2')
    normalized_test_X = preprocessing.normalize(test_X, norm='l2')

    return [train_X.to_numpy(), train_y.to_numpy(), train_X.to_numpy(), train_y.to_numpy(), train_X.to_numpy(), train_y.to_numpy(),
            train_X.to_numpy(),
            test_X.to_numpy(), train_X.to_numpy(), test_X.to_numpy(), normalized_train_X, normalized_test_X,
            test_X.to_numpy(), test_y.to_numpy(), train_X.to_numpy()]


def estimator_tests(configs):
    file_path = configs['machine_learning_data_file_path']
    cluster_data_ini = pd.read_csv(file_path, sep=',', header=0,
                                   names=configs['machine_learning_data_columns'])


    ###############################
    # all options
    ###############################

    classifiers = [MLPClassifier(), SVC(), RandomForestClassifier(), GradientBoostingClassifier(), SGDClassifier(),
                   KNeighborsClassifier(), GaussianNB(), DecisionTreeClassifier(),
                   BaggingClassifier(), ExtraTreesClassifier(), AdaBoostClassifier(),
                   HistGradientBoostingClassifier(), VotingClassifier(estimators=[('rf', RandomForestClassifier()),
                                                                                  ('hgb',
                                                                                   HistGradientBoostingClassifier())],
                                                                      voting='soft')]  # ,
    # GaussianProcessClassifier()]

    labels = ['Neural Network Classifier', 'Support Vector Machine Classifier', 'Random Forest Classifier',
              'Gradiant Boosting Classifier', 'Stochastic Gradient Descent Classifier', 'K Nearest Neighbor Classifier',
              'Gaussian Naive Bayes', 'Decision Tree Classifier', 'Bagging Classifier',
              'Extremely Randomized Tree Classifier', 'Ada Boost Classifier',
              'Histogram Gradient Boosting Classifier', 'Voting Classifier']  # , 'Gaussian Process Classifier']

    target_classes = ['Metric Choice']
    target_labels = ['perfect_metric_choice']

    data_labels = ['All_Features']

    dataset_forms = ['Quantile Transform', 'Standard Scalar', 'Normalized', 'Oversampled', 'Original', 'Undersampled',
                     'Quantile Oversampled']

    #####################################
    # desired options
    ####################################
    classifiers = [MLPClassifier(verbose=True)]

    target_classes = ['Metric Choice']
    target_labels = ['perfect_metric_choice']

    data_labels = ['GRTD', 'GRT', 'GRD', 'GTD', 'RTD', 'GR', 'GT', 'GD', 'RT', 'RD', 'TD', 'G', 'R', 'T', 'D']  #

    dataset_forms = ['Normalized']

    ######################################
    # hyperparameter tuning
    #####################################

    for k, target_label in enumerate(target_labels):
        data = []
        for j, data_label in enumerate(data_labels):
            if data_label == 'GRTD':
                print('GRTD')
                datasets = preprocessing_main(cluster_data_ini.loc[:,
                                              ['number_of_points', 'points_diff', 'z_spread', 'x_spread', 'y_spread',
                                               'pca_prev_diff', 'ransac_prev_diff', 'x_spread_diff', 'y_spread_diff',
                                               'z_spread_diff', 'ransac_pca_diff', 'ransac_pred_diff',
                                               'pca_pred_diff']],
                                              cluster_data_ini[target_label])
            elif data_label == 'GRT':
                print('GRT')
                datasets = preprocessing_main(cluster_data_ini.loc[:,
                                              ['number_of_points', 'z_spread', 'x_spread', 'y_spread',
                                               'pca_prev_diff', 'ransac_prev_diff', 'ransac_pca_diff', 'ransac_pred_diff',
                                               'pca_pred_diff']],
                                              cluster_data_ini[target_label])
            elif data_label == 'GRD':
                print('GRD')
                datasets = preprocessing_main(cluster_data_ini.loc[:,
                                              ['number_of_points', 'points_diff', 'z_spread', 'x_spread', 'y_spread',
                                               'x_spread_diff', 'y_spread_diff',
                                               'z_spread_diff', 'ransac_pca_diff', 'ransac_pred_diff',
                                               'pca_pred_diff']],
                                              cluster_data_ini[target_label])
            elif data_label == 'GTD':
                print('GTD')
                datasets = preprocessing_main(cluster_data_ini.loc[:,
                                              ['number_of_points', 'points_diff', 'z_spread', 'x_spread', 'y_spread',
                                               'pca_prev_diff', 'ransac_prev_diff', 'x_spread_diff', 'y_spread_diff',
                                               'z_spread_diff']],
                                              cluster_data_ini[target_label])
            elif data_label == 'RTD':
                print('RTD')
                datasets = preprocessing_main(cluster_data_ini.loc[:,
                                              ['points_diff',
                                               'pca_prev_diff', 'ransac_prev_diff', 'x_spread_diff', 'y_spread_diff',
                                               'z_spread_diff', 'ransac_pca_diff', 'ransac_pred_diff',
                                               'pca_pred_diff']],
                                              cluster_data_ini[target_label])
            elif data_label == 'GR':
                print('GR')
                datasets = preprocessing_main(cluster_data_ini.loc[:,
                                              ['number_of_points', 'z_spread', 'x_spread', 'y_spread',
                                               'ransac_pca_diff', 'ransac_pred_diff',
                                               'pca_pred_diff']],
                                              cluster_data_ini[target_label])
            elif data_label == 'GT':
                print('GT')
                datasets = preprocessing_main(cluster_data_ini.loc[:,
                                              ['number_of_points', 'z_spread', 'x_spread', 'y_spread',
                                               'pca_prev_diff', 'ransac_prev_diff']],
                                              cluster_data_ini[target_label])
            elif data_label == 'GD':
                print('GD')
                datasets = preprocessing_main(cluster_data_ini.loc[:,
                                              ['number_of_points', 'points_diff', 'z_spread', 'x_spread', 'y_spread',
                                               'x_spread_diff', 'y_spread_diff',
                                               'z_spread_diff']],
                                              cluster_data_ini[target_label])
            elif data_label == 'RT':
                print('RT')
                datasets = preprocessing_main(cluster_data_ini.loc[:,
                                              ['pca_prev_diff', 'ransac_prev_diff', 'ransac_pca_diff', 'ransac_pred_diff',
                                               'pca_pred_diff']],
                                              cluster_data_ini[target_label])
            elif data_label == 'RD':
                print('RD')
                datasets = preprocessing_main(cluster_data_ini.loc[:,
                                              ['points_diff',
                                               'x_spread_diff', 'y_spread_diff',
                                               'z_spread_diff', 'ransac_pca_diff', 'ransac_pred_diff',
                                               'pca_pred_diff']],
                                              cluster_data_ini[target_label])
            elif data_label == 'TD':
                print('TD')
                datasets = preprocessing_main(cluster_data_ini.loc[:,
                                              ['points_diff',
                                               'pca_prev_diff', 'ransac_prev_diff', 'x_spread_diff', 'y_spread_diff',
                                               'z_spread_diff']],
                                              cluster_data_ini[target_label])
            elif data_label == 'G':
                print('G')
                datasets = preprocessing_main(cluster_data_ini.loc[:,
                                              ['number_of_points', 'z_spread', 'x_spread', 'y_spread']],
                                              cluster_data_ini[target_label])
            elif data_label == 'R':
                print('R')
                datasets = preprocessing_main(cluster_data_ini.loc[:,
                                              ['ransac_pca_diff', 'ransac_pred_diff',
                                               'pca_pred_diff']],
                                              cluster_data_ini[target_label])
            elif data_label == 'T':
                print('T')
                datasets = preprocessing_main(cluster_data_ini.loc[:,
                                              ['pca_prev_diff', 'ransac_prev_diff']],
                                              cluster_data_ini[target_label])
            elif data_label == 'D':
                print('D')
                datasets = preprocessing_main(cluster_data_ini.loc[:,
                                              ['points_diff',
                                               'x_spread_diff', 'y_spread_diff',
                                               'z_spread_diff']],
                                              cluster_data_ini[target_label])


            for i, classifier in enumerate(classifiers):
                for n, data_type in enumerate(dataset_forms):
                    if data_type == 'Quantile Transform':
                        train_X = datasets[6]
                        train_y = datasets[1]
                        test_X = datasets[7]
                        test_y = datasets[13]
                    elif data_type == 'Standard Scalar':
                        train_X = datasets[8]
                        train_y = datasets[1]
                        test_X = datasets[9]
                        test_y = datasets[13]
                    elif data_type == 'Normalized':
                        train_X = datasets[10]
                        train_y = datasets[1]
                        test_X = datasets[11]
                        test_y = datasets[13]
                    elif data_type == 'Oversampled':
                        train_X = datasets[2]
                        train_y = datasets[3]
                        test_X = datasets[12]
                        test_y = datasets[13]
                    elif data_type == 'Undersampled':
                        train_X = datasets[4]
                        train_y = datasets[5]
                        test_X = datasets[12]
                        test_y = datasets[13]
                    elif data_type == 'Quantile Oversampled':
                        train_X = datasets[14]
                        train_y = datasets[3]
                        test_X = datasets[7]
                        test_y = datasets[13]
                    else:
                        train_X = datasets[0]
                        train_y = datasets[1]
                        test_X = datasets[12]
                        test_y = datasets[13]

                    print("Training")
                    classifier.fit(train_X, train_y)
                    start = time.time()
                    pred_y = classifier.predict(test_X)
                    print(time.time() - start)
                    class_labels = set(train_y)

                    mat = confusion_matrix(test_y, pred_y, labels=list(class_labels)).ravel()
                    acc = accuracy_score(test_y, pred_y)
                    data.append(round(100 * acc, 2))
                    print('Data type: ' + str(dataset_forms[n]))
                    print('Classifier: ' + str(labels[i]))
                    print('Dataset: ' + str(data_labels[j]))
                    print('Target Class: ' + str(target_classes[k]))
                    print('Accuracy: ' + str(acc))
                    print('Class Labels:' + str(class_labels))
                    print('Confusion Matrix: ' + str(mat) + '\n')
                    
                    joblib.dump(classifier, 'perfect_metric_model' + data_label + '.pkl')
                    print("saved model")
                    print('\n')

    return


with open('configuration.yaml', 'r') as f:
    configs = yaml.safe_load(f)

estimator_tests(configs)
