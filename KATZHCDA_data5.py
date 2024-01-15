
import csv
import math
import random
import h5py
import numpy as np
import pandas as pd
import sortscore
import matplotlib.pyplot as plt
from MakeSimilarityMatrix import MakeSimilarityMatrix
import pickle


def SC(relmatrix, circ_gipsim_matrix):

    return  circ_gipsim_matrix

def SD(relmatrix, dis_gipsim_matrix):

    return dis_gipsim_matrix


if __name__ == '__main__':
    
    # with h5py.File('./Data/disease-circRNA.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_cancer/circRNA_cancer.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_disease_from_circRNADisease/association.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circ2Traits/circRNA_disease.h5','r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    with h5py.File('./Data/circad/circrna_disease.h5', 'r') as hf:
        circrna_disease_matrix = hf['infor'][:]
    
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

    fold = 1
    for i in range(0, len(one_list), split):
        test_index = one_list[i:i + split]
        new_circrna_disease_matrix = circrna_disease_matrix.copy()
    
        for index in test_index:
            new_circrna_disease_matrix[index[0], index[1]] = 0
        roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix
        rel_matrix = new_circrna_disease_matrix
        
        make_sim_matrix = MakeSimilarityMatrix(rel_matrix)
        circ_gipsim_matrix, dis_gipsim_matrix = make_sim_matrix.circsimmatrix, make_sim_matrix.dissimmatrix
        circ_sim_matrix = SC(rel_matrix, circ_gipsim_matrix)
        dis_sim_matrix = SD(rel_matrix, dis_gipsim_matrix)

        prediction_matrix = 0.01*rel_matrix + 0.01**2*(np.dot(circ_sim_matrix, rel_matrix)+ np.dot(rel_matrix, dis_sim_matrix))

        prediction_matrix_real = prediction_matrix.real
        result = pd.DataFrame(prediction_matrix_real)
        np.savetxt("./five_folds_prediction_output/Dataset5/KATZHCDA/KATZHCDA" + str(count) + ".csv", result, delimiter=",")
        count += 1