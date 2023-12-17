
import math
import random
import h5py
import numpy as np
import pandas as pd
import sortscore
import matplotlib.pyplot as plt
from MakeSimilarityMatrix import MakeSimilarityMatrix


def SC(relmatrix, circ_gipsim_matrix):

    return  circ_gipsim_matrix

def SD(relmatrix, dis_gipsim_matrix):

    return dis_gipsim_matrix

def find_key(i, cancer_dict):
    name = list(cancer_dict.keys())[list(cancer_dict.values()).index(i)]

    return name


if __name__ == '__main__':
    # with h5py.File('./Data/disease-circRNA.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    with h5py.File('./Data/circRNA_cancer/circRNA_cancer.h5', 'r') as hf:
        circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_disease_from_circRNADisease/association.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circ2Traits/circRNA_disease.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

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
        if ((False in (new_circrna_disease_matrix[:, i] == 0)) == False):
            continue
        new_circrna_disease_matrix[:, i] = 0

        rel_matrix = new_circrna_disease_matrix
        make_sim_matrix = MakeSimilarityMatrix(rel_matrix)
        circ_gipsim_matrix, dis_gipsim_matrix = make_sim_matrix.circsimmatrix, make_sim_matrix.dissimmatrix
        circ_sim_matrix = SC(rel_matrix, circ_gipsim_matrix)
        dis_sim_matrix = SD(rel_matrix, dis_gipsim_matrix)

        prediction_matrix = 0.01*rel_matrix + 0.01**2*(np.dot(circ_sim_matrix, rel_matrix)+ np.dot(rel_matrix, dis_sim_matrix))

    prediction_matrix_real = prediction_matrix.real
    result = pd.DataFrame(prediction_matrix_real)
    result
    np.savetxt("./denovo_prediction_output/Dataset2/KATZHCDA_result_data2.csv", result, delimiter=",")
