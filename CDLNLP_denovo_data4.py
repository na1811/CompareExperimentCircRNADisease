
import math
import random
import h5py
import numpy as np
import LNLP_method
import sortscore
import matplotlib.pyplot as plt
import pandas as pd
from MakeSimilarityMatrix import MakeSimilarityMatrix


def find_key(i, cancer_dict):
    name = list(cancer_dict.keys())[list(cancer_dict.values()).index(i)]

    return name


if __name__ == '__main__':
    alpha = 0.1
    neighbor_rate = 0.9
    weight = 1.0

    # with h5py.File('./Data/disease-circRNA.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_cancer/circRNA_cancer.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_disease_from_circRNADisease/association.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    with h5py.File('./Data/circ2Traits/circRNA_disease.h5', 'r') as hf:
        circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circad/circrna_disease.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]


    all_tpr = []
    all_fpr = []
    all_recall = []
    all_precision = []
    all_accuracy = []
    all_F1 = []

    # cancer_dict = {'glioma': 7, 'bladder cancer':9, 'breast cancer': 10,'cervical cancer': 53,'cervical carcinoma': 64,'colorectal cancer':11,'gastric cancer':19}

    # cancer_dict = {'glioma': 23, 'bladder cancer': 2, 'breast cancer': 4, 'cervical cancer': 6,
    #                 'colorectal cancer': 12, 'gastric cancer': 20}

    # cancer_dict = {'glioma': 20, 'bladder cancer': 19, 'breast cancer': 6, 'cervical cancer': 16,
    #                 'colorectal cancer': 1, 'gastric cancer': 0}

    # # circ2Traits
    cancer_dict = {'bladder cancer': 58, 'breast cancer':46, 'glioma':89, 'glioblastoma':88,
                   'glioblastoma multiforme':59, 'cervical cancer':23, 'colorectal cancer':6, 'gastric cancer':15}

    # circad
    # cancer_dict = {'bladder cancer':94, 'breast cancer':53, 'triple-negative breast cancer':111, 'gliomas':56, 'glioma':76,
    #                 'cervical cancer':65, 'colorectal cancer':143, 'gastric cancer':28}

    # denovo start
    for i in range(circrna_disease_matrix.shape[1]):
        new_circrna_disease_matrix = circrna_disease_matrix.copy()
        roc_circrna_disease_matrix = circrna_disease_matrix.copy()
        if ((False in (new_circrna_disease_matrix[:,i]==0))==False):
            continue
        new_circrna_disease_matrix[:,i] = 0

        rel_matrix = new_circrna_disease_matrix


        prediction_matrix = LNLP_method.linear_neighbor_predict(rel_matrix, alpha, neighbor_rate, weight)
        prediction_matrix = prediction_matrix.A

    prediction_matrix_real = prediction_matrix.real
    result = pd.DataFrame(prediction_matrix_real)
    result
    np.savetxt("./denovo_output/Dataset4/CDLNLP_result_data4.csv", result, delimiter=",")
