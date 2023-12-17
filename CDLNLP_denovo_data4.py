
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
    np.savetxt("./denovo_prediction_output/Dataset4/CDLNLP_result_data4.csv", result, delimiter=",")
