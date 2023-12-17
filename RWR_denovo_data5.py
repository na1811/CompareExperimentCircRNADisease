import math
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sortscore
from MakeSimilarityMatrix import MakeSimilarityMatrix
import sys
import pandas as pd

sys.setrecursionlimit(9000000)

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

def find_key(i, cancer_dict):
    name = list(cancer_dict.keys())[list(cancer_dict.values()).index(i)]

    return name

if __name__ == '__main__':
    
    # with h5py.File('./Data/disease-circRNA.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_cancer/circRNA_cancer.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_disease_from_circRNADisease/association.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circ2Traits/circRNA_disease.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    with h5py.File('./Data/circad/circrna_disease.h5', 'r') as hf:
        circrna_disease_matrix = hf['infor'][:]

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

        make_sim_matrix = MakeSimilarityMatrix(rel_matrix)
        circ_gipsim_matrix, dis_gipsim_matrix = make_sim_matrix.circsimmatrix, make_sim_matrix.dissimmatrix

        A_matrix = rel_matrix.T
        WDD_matrix = WDD(A_matrix,  dis_gipsim_matrix, circnum, disnum, 0.7)
        WCC_matrix = WCC(A_matrix.T, circ_gipsim_matrix, circnum, disnum, 0.7)
        WTD_matrix = WTD(A_matrix, circnum, disnum, 0.7)
        WTC_matrix = WTC(A_matrix.T, circnum, disnum, 0.7)
        W_matrix = np.vstack((np.hstack((WDD_matrix, WTD_matrix)), np.hstack((WTC_matrix, WCC_matrix))))

        p0Matrix = np.zeros((disnum+circnum, disnum))
        for m in range(disnum):
            itemCircRNANum = np.sum(A_matrix[m,:])
            for n in range(circnum):
                if(A_matrix[m,n] == 1):
                    p0Matrix[disnum+n,m] = 1.0 / itemCircRNANum

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
    result
    np.savetxt("./denovo_prediction_output/Dataset5/RWR_result_data5.csv", result, delimiter=",")
