
import csv
import math
import random
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sortscore
import pickle
from MakeSimilarityMatrix import MakeSimilarityMatrix

def WDD(A_matrix, dis_gipsim_matrix, circnum, disnum, paramater=0.5):
    WDDmatrix = np.zeros((disnum, disnum))
    for i in range(disnum):
        Asum = np.sum(A_matrix[i,:])
        Dsum = np.sum(dis_gipsim_matrix[i,:])
        for j in range(disnum):
            if Asum==0:
                WDDmatrix[i,j] = dis_gipsim_matrix[i,j] / Dsum
            else:
                WDDmatrix[i,j] = ((1 - paramater) * dis_gipsim_matrix[i,j]) / Dsum
    return WDDmatrix

def WCC(A_matrix, circ_gipsim_matrix, circnum, disnum, parameter=0.5):
    WCCmatrix =np.zeros((circnum, circnum))
    for i in range(circnum):
        Asum = np.sum(A_matrix[i,:])
        Csum = np.sum(circ_gipsim_matrix[i,:])
        for j in range(circnum):
            if Asum==0:
                WCCmatrix[i,j] = circ_gipsim_matrix[i,j] / Csum
            else:
                WCCmatrix[i,j] = ((1 - parameter) * circ_gipsim_matrix[i,j]) / Csum
    return WCCmatrix

def WTD(A_matrix, circnum, disnum, parameter=0.5):
    WTDmatrix = np.zeros((disnum, circnum))
    for i in range(disnum):
        Asum = np.sum(A_matrix[i,:])
        for j in range(circnum):
            if(Asum == 0):
                WTDmatrix[i,j] = 0
            else:
                WTDmatrix[i,j] = parameter * A_matrix[i,j] / Asum
    return WTDmatrix

def WTC(A_matrix, circnum, disnum, parameter=0.5):
    WTCmatrix = np.zeros((circnum, disnum))
    for i in range(circnum):
        Asum = np.sum(A_matrix[i,:])
        for j in range(disnum):
            if(Asum == 0):
                WTCmatrix[i,j] = 0
            else:
                WTCmatrix[i,j] = parameter * A_matrix[i,j] / Asum
    return WTCmatrix

def p(WMatrix, t, p0Matrix, r=0.5):
    if(t == 0):
        return p0Matrix
    else:
        return (1-r) * np.matmul(WMatrix, p(WMatrix,t-1,p0Matrix)) + r*p0Matrix

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

    # 5-fold start
    fold = 1
    for i in range(0, len(one_list), split):
        test_index = one_list[i:i + split]
        train_index = list(set(one_list) - set(test_index))
        new_circrna_disease_matrix = circrna_disease_matrix.copy()

        for index in test_index:
            new_circrna_disease_matrix[index[0], index[1]] = 0
        roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix
        rel_matrix = new_circrna_disease_matrix
        circnum = rel_matrix.shape[0]
        disnum = rel_matrix.shape[1]

        make_sim_matrix = MakeSimilarityMatrix(rel_matrix)
        circ_gipsim_matrix, dis_gipsim_matrix = make_sim_matrix.circsimmatrix, make_sim_matrix.dissimmatrix

        A_matrix = rel_matrix.T
        WDD_matrix = WDD(A_matrix,  dis_gipsim_matrix, circnum, disnum, 0.7)
        WCC_matrix = WCC(A_matrix.T, circ_gipsim_matrix, circnum, disnum, 0.7)
        WTD_matrix = WTD(A_matrix, circnum, disnum, 0.7)
        WTC_matrix = WTC(A_matrix.T, circnum, disnum, 0.7)
        W_matrix = np.vstack((np.hstack((WDD_matrix, WTD_matrix)), np.hstack((WTC_matrix, WCC_matrix))))

        p0Matrix = np.zeros((disnum+circnum, disnum))
        for i in range(disnum):
            itemCircRNANum = np.sum(A_matrix[i,:])
            for j in range(circnum):
                if(A_matrix[i,j] == 1):
                    p0Matrix[disnum+j,i] = 1.0 / itemCircRNANum

        t=1
        pt = p0Matrix
        circRNAdisNum = circnum * disnum

        while(True):
            pted = p(W_matrix,t,p0Matrix)
            Delta = abs(pted - pt)
            if(np.sum(Delta) / circRNAdisNum < 1e-6):
                break
            pt = pted
            t += 1

        prediction_matrix = pted[disnum:,:]
        prediction_matrix_real = prediction_matrix.real
        result = pd.DataFrame(prediction_matrix_real)
        np.savetxt("./five_folds_prediction_output/Dataset5/RWR/RWR" + str(count) + ".csv", result, delimiter=",")
        count += 1
