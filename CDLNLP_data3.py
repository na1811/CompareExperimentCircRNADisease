
import csv
import math
import random
import h5py
import numpy as np
import pandas as pd
import pickle
import LNLP_method
import sortscore
import matplotlib.pyplot as plt
from MakeSimilarityMatrix import MakeSimilarityMatrix

if __name__ == '__main__':
    alpha = 0.1
    neighbor_rate = 0.9
    weight = 1.0

    # with h5py.File('./Data/disease-circRNA.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_cancer/circRNA_cancer.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    with h5py.File('./Data/circRNA_disease_from_circRNADisease/association.h5', 'r') as hf:
        circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circ2Traits/circRNA_disease.h5','r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circad/circrna_disease.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]
    
    file_name = "saved_variables_data3.pkl"
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

    # 5-fold start
    fold = 1
    for i in range(0, len(one_list), split):
        test_index = one_list[i:i + split]
        train_index = list(set(one_list)-set(test_index))
        new_circrna_disease_matrix = circrna_disease_matrix.copy()

        for index in test_index:
            new_circrna_disease_matrix[index[0], index[1]] = 0
        roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix
        rel_matrix = new_circrna_disease_matrix

        prediction_matrix = LNLP_method.linear_neighbor_predict(rel_matrix, alpha, neighbor_rate, weight)
        prediction_matrix = prediction_matrix.A

    prediction_matrix_real = prediction_matrix.real
    result = pd.DataFrame(prediction_matrix_real)
    result
    np.savetxt("./five_folds_prediction_output/Dataset3/CDLNLP_result_data3.csv", result, delimiter=",")
