import numpy as np
import h5py
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import sortscore
# import sklearn.cluster as sc
from MakeSimilarityMatrix import MakeSimilarityMatrix#计算高斯核相似性

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

import time

from sklearn.preprocessing import minmax_scale
import pandas as pd
import scipy.io
import pickle

# Edit comment for datasets you want to run

# Dataset1
# with h5py.File('./Data/disease-circRNA.h5', 'r') as hf:
#     circrna_disease_matrix = hf['infor'][:]

# Dataset2
# with h5py.File('./Data/circRNA_cancer/circRNA_cancer.h5', 'r') as hf:
#     circrna_disease_matrix = hf['infor'][:]

# Dataset3
# with h5py.File('./Data/circRNA_disease_from_circRNADisease/association.h5', 'r') as hf:
#     circrna_disease_matrix = hf['infor'][:]

# Dataset4
# with h5py.File('./Data/circ2Traits/circRNA_disease.h5','r') as hf:
#     circrna_disease_matrix = hf['infor'][:]

# Dataset5
with h5py.File('./Data/circad/circrna_disease.h5', 'r') as hf:
    circrna_disease_matrix = hf['infor'][:]

# Edit comment for datasets you want to run
# Dataset1
#file_name = "saved_variables_data1.pkl"

# Dataset2
#file_name = "saved_variables_data2.pkl"

# Dataset3
#file_name = "saved_variables_data3.pkl"

# Dataset4
#file_name = "saved_variables_data4.pkl"

# Dataset5
file_name = "saved_variables_data5.pkl"

with open(file_name, 'rb') as file:
    loaded_variables = pickle.load(file)
index_tuple = loaded_variables["index_tuple"]
one_list = loaded_variables["one_list"]
split = loaded_variables["split"]

all_tpr = []
all_fpr = []
all_recall = []
all_precision = []
all_accuracy = []
all_F1 = []

count = 1
# 5-fold start
for i in range(0, len(one_list), split):
    test_index = one_list[i:i + split]
    new_circrna_disease_matrix = circrna_disease_matrix.copy()
    
    # Edit comment for datasets you want to run
    
    # Dataset1
    # rnmflp_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset1/RNMFLP/RNMFLP" + str(count) + ".csv", delimiter=',')
    # katzh_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset1/KATHZHCDA/KATHZHCDA" + str(count) + ".csv", delimiter=',')
    # gmnn_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset1/GMNN2CD/GMNN2CD" + str(count) + ".csv", delimiter=',')
    # rwr_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset1/RWR/RWR" + str(count) + ".csv", delimiter=',')
    # cdlnlp_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset1/CDLNLP/CDLNLP" + str(count) + ".csv", delimiter=',')
    
    # Dataset2
    # rnmflp_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset2/RNMFLP/RNMFLP" + str(count) + ".csv", delimiter=',')
    # katzh_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset2/KATHZHCDA/KATHZHCDA" + str(count) + ".csv", delimiter=',')
    # gmnn_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset2/GMNN2CD/GMNN2CD" + str(count) + ".csv", delimiter=',')
    # rwr_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset2/RWR/RWR" + str(count) + ".csv", delimiter=',')
    # cdlnlp_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset2/CDLNLP/CDLNLP" + str(count) + ".csv", delimiter=',')
    
    # Dataset3
    # rnmflp_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset3/RNMFLP/RNMFLP" + str(count) + ".csv", delimiter=',')
    # katzh_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset3/KATHZHCDA/KATHZHCDA" + str(count) + ".csv", delimiter=',')
    # gmnn_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset3/GMNN2CD/GMNN2CD" + str(count) + ".csv", delimiter=',')
    # rwr_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset3/RWR/RWR" + str(count) + ".csv", delimiter=',')
    # cdlnlp_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset3/CDLNLP/CDLNLP" + str(count) + ".csv", delimiter=',')
    
    # Dataset4
    # rnmflp_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset4/RNMFLP/RNMFLP" + str(count) + ".csv", delimiter=',')
    # katzh_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset4/KATHZHCDA/KATHZHCDA" + str(count) + ".csv", delimiter=',')
    # gmnn_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset4/GMNN2CD/GMNN2CD" + str(count) + ".csv", delimiter=',')
    # rwr_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset4/RWR/RWR" + str(count) + ".csv", delimiter=',')
    # cdlnlp_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset4/CDLNLP/CDLNLP" + str(count) + ".csv", delimiter=',')
    
    # Dataset5
    rnmflp_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset5/RNMFLP/RNMFLP" + str(count) + ".csv", delimiter=',')
    katzh_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset5/KATHZHCDA/KATHZHCDA" + str(count) + ".csv", delimiter=',')
    gmnn_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset5/GMNN2CD/GMNN2CD" + str(count) + ".csv", delimiter=',')
    rwr_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset5/RWR/RWR" + str(count) + ".csv", delimiter=',')
    cdlnlp_matrix = np.genfromtxt("./five_folds_prediction_output/Dataset5/CDLNLP/CDLNLP" + str(count) + ".csv", delimiter=',')
    
    prediction_matrix = (gmnn_matrix + rwr_matrix + katzh_matrix + cdlnlp_matrix + rnmflp_matrix)/5
    count += 1
    
    for index in test_index:
        new_circrna_disease_matrix[index[0], index[1]] = 0
    roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix
    rel_matrix = new_circrna_disease_matrix
    aa = prediction_matrix.shape
    bb = roc_circrna_disease_matrix.shape
    zero_matrix = np.zeros((prediction_matrix.shape[0], prediction_matrix.shape[1]))

    score_matrix_temp = prediction_matrix.copy()
    score_matrix = score_matrix_temp + zero_matrix
    minvalue = np.min(score_matrix)
    score_matrix[np.where(roc_circrna_disease_matrix == 2)] = minvalue - 20#？
    sorted_circrna_disease_matrix, sorted_score_matrix, sort_index = sortscore.sort_matrix(score_matrix,roc_circrna_disease_matrix)

    tpr_list = []
    fpr_list = []
    recall_list = []
    precision_list = []
    accuracy_list = []
    F1_list = []
    for cutoff in range(sorted_circrna_disease_matrix.shape[0]):
        P_matrix = sorted_circrna_disease_matrix[0:cutoff + 1, :]
        N_matrix = sorted_circrna_disease_matrix[cutoff + 1:sorted_circrna_disease_matrix.shape[0] + 1, :]
        TP = np.sum(P_matrix == 1)
        FP = np.sum(P_matrix == 0)
        TN = np.sum(N_matrix == 0)
        FN = np.sum(N_matrix == 1)
        tpr = TP / (TP + FN)
        fpr = FP / (FP + TN)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        recall_list.append(recall)
        precision_list.append(precision)
        accuracy = (TN + TP) / (TN + TP + FN + FP)
        F1 = (2 * TP) / (2*TP + FP + FN)
        if (2*TP + FP + FN)==0 :
            F1 = 0
        F1_list.append(F1)
        accuracy_list.append(accuracy)

    tpr_arr_epoch = np.array(tpr_list)
    fpr_arr_epoch = np.array(fpr_list)
    recall_arr_epoch = np.array(recall_list)
    precision_arr_epoch = np.array(precision_list)
    accuracy_arr_epoch = np.array(accuracy_list)
    F1_arr_epoch = np.array(F1_list)
    
    all_tpr.append(tpr_list)
    all_fpr.append(fpr_list)
    all_recall.append(recall_list)
    all_precision.append(precision_list)
    all_accuracy.append(accuracy_list)
    all_F1.append(F1_list)

tpr_arr = np.array(all_tpr)
fpr_arr = np.array(all_fpr)
recall_arr = np.array(all_recall)
precision_arr = np.array(all_precision)
accuracy_arr = np.array(all_accuracy)
F1_arr = np.array(all_F1)

mean_cross_tpr = np.mean(tpr_arr, axis=0)  # axis=0
mean_cross_fpr = np.mean(fpr_arr, axis=0)
mean_cross_recall = np.mean(recall_arr, axis=0)
mean_cross_precision = np.mean(precision_arr, axis=0)
mean_cross_accuracy = np.mean(accuracy_arr, axis=0)
mean_accuracy = np.mean(np.mean(accuracy_arr, axis=1), axis=0)
mean_recall = np.mean(np.mean(recall_arr, axis=1), axis=0)
mean_precision = np.mean(np.mean(precision_arr, axis=1), axis=0)
mean_F1 = np.mean(np.mean(F1_arr, axis=1), axis=0)
print("accuracy:%.4f,recall:%.4f,precision:%.4f,F1:%.4f"%(mean_accuracy, mean_recall, mean_precision, mean_F1))

roc_auc = np.trapz(mean_cross_tpr, mean_cross_fpr)
AUPR = np.trapz(mean_cross_precision, mean_cross_recall)

print("AUC:%.4f,AUPR:%.4f"%(roc_auc, AUPR))

with h5py.File('./PlotFigure/GMNN2CD_circRNADisease_10fold_AUC.h5','w') as hf:
    hf['fpr'] = mean_cross_fpr
    hf['tpr'] = mean_cross_tpr
with h5py.File('./PlotFigure/GMNN2CD_circRNADisease_10fold_AUPR.h5','w') as h:
    h['recall'] = mean_cross_recall
    h['precision'] = mean_cross_precision

plt.plot(mean_cross_fpr, mean_cross_tpr, label='mean ROC=%0.4f' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0)
plt.savefig("./FinalResultPng/roc-circRNADisease_10fold.png")
plt.show()


