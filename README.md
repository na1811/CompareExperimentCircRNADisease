# Five methods: RWR, CDLNLP, GMNN2CD, KATZHCDA, RNMFLP

# Run five-folds (Update 15/01/2024)
## Step 1: Run each method with each dataset to get predict_matrix 
+ The way to run the following 4 methods is similar: RWR, CDLNLP, GMNN2CD, KATZHCDA
  
If you want to run RWR with Dataset 1, use the command: bsub -o RWR_data1.output -e RWR_data1.error -n 8 "conda run -n base python RWR_data1.py". 

Note that before running, you need to check that you have the correct path to store prediction_matrix. For example, when you run the file "RWR_data1.py", the following output storage path is needed: 
"./five_folds_prediction_output/Dataset1/RWR". Please check the code file "RWR_data1.py" for more clarity!

After that, predict_matrix will be saved into 5 files corresponding to the folds according to the following directory path: "./5_folds_prediction_output/Dataset1/RWR"

Similarly, for each method and each different dataset, you need to run the following commands:

Run RWR with 5 datasets as follows: 
  + bsub -o RWR_data1.output -e RWR_data1.error -n 8 "conda run -n base python RWR_data1.py"
  + bsub -o RWR_data2.output -e RWR_data2.error -n 8 "conda run -n base python RWR_data2.py"
  + bsub -o RWR_data3.output -e RWR_data3.error -n 8 "conda run -n base python RWR_data3.py"
  + bsub -o RWR_data4.output -e RWR_data4.error -n 8 "conda run -n base python RWR_data4.py"
  + bsub -o RWR_data5.output -e RWR_data5.error -n 8 "conda run -n base python RWR_data5.py"

Run CDLNLP with 5 datasets as follows: 
  + bsub -o CDLNLP_data1.output -e CDLNLP_data1.error -n 8 "conda run -n base python CDLNLP_data1.py"
  + bsub -o CDLNLP_data2.output -e CDLNLP_data2.error -n 8 "conda run -n base python CDLNLP_data2.py"
  + bsub -o CDLNLP_data3.output -e CDLNLP_data3.error -n 8 "conda run -n base python CDLNLP_data3.py"
  + bsub -o CDLNLP_data4.output -e CDLNLP_data4.error -n 8 "conda run -n base python CDLNLP_data4.py"
  + bsub -o CDLNLP_data5.output -e CDLNLP_data5.error -n 8 "conda run -n base python CDLNLP_data5.py"

Run GMNN2CD with 5 datasets as follows: 
  + bsub -o GMNN2CD_data1.output -e GMNN2CD_data1.error -n 8 "conda run -n base python GMNN2CD_data1.py"
  + bsub -o GMNN2CD_data2.output -e GMNN2CD_data2.error -n 8 "conda run -n base python GMNN2CD_data2.py"
  + bsub -o GMNN2CD_data3.output -e GMNN2CD_data3.error -n 8 "conda run -n base python GMNN2CD_data3.py"
  + bsub -o GMNN2CD_data4.output -e GMNN2CD_data4.error -n 8 "conda run -n base python GMNN2CD_data4.py"
  + bsub -o GMNN2CD_data5.output -e GMNN2CD_data5.error -n 8 "conda run -n base python GMNN2CD_data5.py"

Run KATZHCDA with 5 datasets as follows: 
  + bsub -o KATZHCDA_data1.output -e KATZHCDA_data1.error -n 8 "conda run -n base python KATZHCDA_data1.py"
  + bsub -o KATZHCDA_data2.output -e KATZHCDA_data2.error -n 8 "conda run -n base python KATZHCDA_data2.py"
  + bsub -o KATZHCDA_data3.output -e KATZHCDA_data3.error -n 8 "conda run -n base python KATZHCDA_data3.py"
  + bsub -o KATZHCDA_data4.output -e KATZHCDA_data4.error -n 8 "conda run -n base python KATZHCDA_data4.py"
  + bsub -o KATZHCDA_data5.output -e KATZHCDA_data5.error -n 8 "conda run -n base python KATZHCDA_data5.py"

The way run RNMFLP method:
+ Go to "RNMFLP main" folder, run file "fivefols.m"
  Note that for each different Dataset, you need to edit the comments in the file "fivefols.m" to run the correct dataset.
  Then, save each predict_scores matrix with the names RNMFLP1, RNMFLP2, RNMFLP3, RNMFLP4, RNMFLP5 respectively according to the folds and corresponding to each data set into the path 
  "./five_folds_prediction_output/Dataset{dataset}/RNMFLP/RNMFLP{fold}.csv" for step 2

## Step 2: Combine 5 predict_matrix for evaluation 
Run file "IntegratedApproachFivefolds.py" => The result will be printed
Note that for each different Dataset, you need to edit the comments in the file "IntegratedApproachFivefolds.py" to run the correct dataset.


# Run denovo
## Step 1: Run each method with each dataset to get predict_matrix (Similar to five-folds)
+ The way to run the following 4 methods is similar: RWR, CDLNLP, GMNN2CD, KATZHCDA
  
If you want to run RWR with Dataset 1, use the command: bsub -o RWR_denovo_data1.output -e RWR_denovo_data1.error -n 8 "conda run -n base python RWR_denovo_data1.py"

Then, predict_matrix output will be saved as path: "./denovo_prediction_output/Dataset1/RWR_result_data1.csv"

Similarly, for each method and each different dataset, you need to run the following commands:

Run RWR with 5 datasets as follows: 
  + bsub -o RWR_denovo_data1.output -e RWR_denovo_data1.error -n 8 "conda run -n base python RWR_denovo_data1.py"
  + bsub -o RWR_denovo_data2.output -e RWR_denovo_data2.error -n 8 "conda run -n base python RWR_denovo_data2.py"
  + bsub -o RWR_denovo_data3.output -e RWR_denovo_data3.error -n 8 "conda run -n base python RWR_denovo_data3.py"
  + bsub -o RWR_denovo_data4.output -e RWR_denovo_data4.error -n 8 "conda run -n base python RWR_denovo_data4.py"
  + bsub -o RWR_denovo_data5.output -e RWR_denovo_data5.error -n 8 "conda run -n base python RWR_denovo_data5.py"

Run CDLNLP with 5 datasets as follows: 
  + bsub -o CDLNLP_denovo_data1.output -e CDLNLP_denovo_data1.error -n 8 "conda run -n base python CDLNLP_denovo_data1.py"
  + bsub -o CDLNLP_denovo_data2.output -e CDLNLP_denovo_data2.error -n 8 "conda run -n base python CDLNLP_denovo_data2.py"
  + bsub -o CDLNLP_denovo_data3.output -e CDLNLP_denovo_data3.error -n 8 "conda run -n base python CDLNLP_denovo_data3.py"
  + bsub -o CDLNLP_denovo_data4.output -e CDLNLP_denovo_data4.error -n 8 "conda run -n base python CDLNLP_denovo_data4.py"
  + bsub -o CDLNLP_denovo_data5.output -e CDLNLP_denovo_data5.error -n 8 "conda run -n base python CDLNLP_denovo_data5.py"

Run GMNN2CD with 5 datasets as follows: 
  + bsub -o GMNN2CD_denovo_data1.output -e GMNN2CD_denovo_data1.error -n 8 "conda run -n base python GMNN2CD_denovo_data1.py"
  + bsub -o GMNN2CD_denovo_data2.output -e GMNN2CD_denovo_data2.error -n 8 "conda run -n base python GMNN2CD_denovo_data2.py"
  + bsub -o GMNN2CD_denovo_data3.output -e GMNN2CD_denovo_data3.error -n 8 "conda run -n base python GMNN2CD_denovo_data3.py"
  + bsub -o GMNN2CD_denovo_data4.output -e GMNN2CD_denovo_data4.error -n 8 "conda run -n base python GMNN2CD_denovo_data4.py"
  + bsub -o GMNN2CD_denovo_data5.output -e GMNN2CD_denovo_data5.error -n 8 "conda run -n base python GMNN2CD_denovo_data5.py"

Run KATZHCDA with 5 datasets as follows: 
  + bsub -o KATZHCDA_denovo_data1.output -e KATZHCDA_denovo_data1.error -n 8 "conda run -n base python KATZHCDA_denovo_data1.py"
  + bsub -o KATZHCDA_denovo_data2.output -e KATZHCDA_denovo_data2.error -n 8 "conda run -n base python KATZHCDA_denovo_data2.py"
  + bsub -o KATZHCDA_denovo_data3.output -e KATZHCDA_denovo_data3.error -n 8 "conda run -n base python KATZHCDA_denovo_data3.py"
  + bsub -o KATZHCDA_denovo_data4.output -e KATZHCDA_denovo_data4.error -n 8 "conda run -n base python KATZHCDA_denovo_data4.py"
  + bsub -o KATZHCDA_denovo_data5.output -e KATZHCDA_denovo_data5.error -n 8 "conda run -n base python KATZHCDA_denovo_data5.py"

The way run RNMFLP method:
+ Go to "RNMFLP main" folder, run file "denovo.m"
  Note that for each different Dataset, you need to edit the comments in the file "denovo.m" to run the correct dataset.
  Then, save each predict_scores matrix corresponding to each dataset into the path "./denovo_prediction_output/Dataset{dataset}/RNMFLP_result_data{dataset}.csv" for step 2

## Step 2: Combine 5 predict_matrix for evaluation 
Run file "IntegratedApproachDenovo.py" => The result will be printed
Note that for each different Dataset, you need to edit the comments in the file "IntegratedApproachDenovo.py" to run the correct dataset.
