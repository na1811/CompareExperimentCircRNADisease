clear;
seed = 12345678;
rand('seed', seed);
nfolds = 5; nruns=1;
dataname = '';

% import dataset
% Dataset1
y = importdata('associations1.xls');

% Dataset2
%y = importdata('associations2.xls');

% Dataset3
%y = importdata('associations3.xls');

% Dataset4
%y = importdata('associations4.xls');

% Dataset5
%y = importdata('associations5.xls');

% y = h5read('./Data/circRNA_disease_from_circRNADisease/association.h5','/g4/lat');
fold_aupr=[];fold_auc=[];fold_accuracy=[];fold_precision=[];fold_recall=[];fold_F1=[];
fold_tpr=[];fold_fpr=[];

for run=1:nruns
    % split folds
	[num_D,num_G] = size(y)
	crossval_idx = crossvalind('Kfold',y(:),nfolds);
   

    for fold=1:nfolds
        train_idx = find(crossval_idx~=fold);
        test_idx  = find(crossval_idx==fold);

        y_train = y;
        
% 		y_train(test_idx) = 0;
        		
        %% 4. evaluate predictions
        yy=y;

		
		test_labels = yy(test_idx);
%         [a,b]=size(test_idx);
		predict_scores = RNMFLP_predict(y_train,80,0.001,0.7,4);    
    
    end
end
% RMSE = sqrt(sum((globa_predict_y_lp-globa_true_y_lp).^2)/length(globa_predict_y_lp))
% mean_accurary = mean(fold_accurary)
% mean_precision = mean(fold_precision,1);
% mean_recall = mean(fold_recall,1);
% mean_tpr = mean(fold_tpr,1);
% mean_fpr = mean(fold_fpr,1);
% roc_auc = trapz(mean_fpr, mean_tpr)
% aupr_auc = trapz(mean_recall,mean_precision)
% mean_accuracy = mean(mean(fold_accuracy, 2), 1)
% mean_recall = mean(mean(fold_recall, 2), 1)
% mean_precision = mean(mean(fold_precision, 2), 1)
% mean_F1 = mean(mean(fold_F1, 2), 1)

% dlmwrite('data1_10_fold_fpr.txt',mean_fpr,'delimiter', '\t');
% dlmwrite('data1_10_fold_tpr.txt',mean_tpr,'delimiter', '\t');
% dlmwrite('data1_10_fold_recall.txt',mean_recall,'delimiter', '\t');
% dlmwrite('data1_10_fold_precision.txt',mean_precision,'delimiter', '\t');
% mean_aupr = mean(fold_aupr)
% mean_auc = mean(fold_auc)
% mean_accurary = mean(fold_accurary)
% mean_precision = mean(fold_precision)
% mean_recall = mean(fold_recall)
% mean_f1 = mean(fold_F1)
% mean_tpr = mean(fold_tpr)
% mean_fpr = mean(fold_fpr)