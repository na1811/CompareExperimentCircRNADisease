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
with h5py.File('./Data/circ2Traits/circRNA_disease.h5','r') as hf:
    circrna_disease_matrix = hf['infor'][:]

# Dataset5
# with h5py.File('./Data/circad/circrna_disease.h5', 'r') as hf:
#     circrna_disease_matrix = hf['infor'][:]
    
# Đọc dữ liệu từ các file CSV
rnmflp_matrix = np.genfromtxt('./denovo_prediction_output/Dataset4/RNMFLP_result_data4.csv', delimiter=',')
katzh_matrix = np.genfromtxt('./denovo_prediction_output/Dataset4/KATZHCDA_result_data4.csv', delimiter=',')
gmnn_matrix = np.genfromtxt('./denovo_prediction_output/Dataset4/GMNN2CD_result_data4.csv', delimiter=',')
rwr_matrix = np.genfromtxt('./denovo_prediction_output/Dataset4/RWR_result_data4.csv', delimiter=',')
cdlnlp_matrix = np.genfromtxt('./denovo_prediction_output/Dataset4/CDLNLP_result_data4.csv', delimiter=',')

prediction_matrix = (gmnn_matrix + rwr_matrix + rnmflp_matrix + katzh_matrix + cdlnlp_matrix)/5 

all_tpr = []
all_fpr = []
all_recall = []
all_precision = []
all_accuracy = []
all_F1 = []

# denovo start
for i in range(circrna_disease_matrix.shape[1]):
    new_circrna_disease_matrix = circrna_disease_matrix.copy()
    roc_circrna_disease_matrix = circrna_disease_matrix.copy()
    if ((False in (new_circrna_disease_matrix[:, i] == 0)) == False):
        continue
    new_circrna_disease_matrix[:, i] = 0
    rel_matrix = new_circrna_disease_matrix
    circnum = rel_matrix.shape[0]
    disnum = rel_matrix.shape[1]
    
    aa = prediction_matrix.shape
    bb = roc_circrna_disease_matrix.shape
    zero_matrix = np.zeros((prediction_matrix.shape[0], prediction_matrix.shape[1]))

    sort_index = np.argsort(-prediction_matrix[:, i], axis=0)
    sorted_circrna_disease_row = roc_circrna_disease_matrix[:, i][sort_index]

    tpr_list = []
    fpr_list = []
    recall_list = []
    precision_list = []
    accuracy_list = []
    F1_list = []

    for cutoff in range(1, rel_matrix.shape[0] + 1):
        P_vector = sorted_circrna_disease_row[0:cutoff]
        N_vector = sorted_circrna_disease_row[cutoff:]
        TP = np.sum(P_vector == 1)
        FP = np.sum(P_vector == 0)
        TN = np.sum(N_vector == 0)
        FN = np.sum(N_vector == 1)
        tpr = TP / (TP + FN)
        fpr = FP / (FP + TN)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        F1 = (2 * TP) / (2 * TP + FP + FN)
        F1_list.append(F1)
        recall_list.append(recall)
        precision_list.append(precision)
        accuracy = (TN + TP) / (TN + TP + FN + FP)
        accuracy_list.append(accuracy)

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

mean_denovo_tpr = np.mean(tpr_arr, axis=0) 
mean_denovo_fpr = np.mean(fpr_arr, axis=0)
mean_denovo_recall = np.mean(recall_arr, axis=0)
mean_denovo_precision = np.mean(precision_arr, axis=0)
mean_denovo_accuracy = np.mean(accuracy_arr, axis=0)

mean_accuracy = np.mean(np.mean(accuracy_arr, axis=1), axis=0)
mean_recall = np.mean(np.mean(recall_arr, axis=1), axis=0)
mean_precision = np.mean(np.mean(precision_arr, axis=1), axis=0)
mean_F1 = np.mean(np.mean(F1_arr, axis=1), axis=0)
print("accuracy:%.4f,recall:%.4f,precision:%.4f,F1:%.4f" % (mean_accuracy, mean_recall, mean_precision, mean_F1))

roc_auc = np.trapz(mean_denovo_tpr, mean_denovo_fpr)
AUPR = np.trapz(mean_denovo_precision, mean_denovo_recall)
print("AUC:%.4f,AUPR:%.4f" % (roc_auc, AUPR))

plt.plot(mean_denovo_fpr, mean_denovo_tpr, label='mean ROC=%0.4f' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0)
plt.savefig("./FinalResultPng/roc-RWR_small_circ2Traits_denovo.png")
print("runtime over, now is :")
plt.show()
