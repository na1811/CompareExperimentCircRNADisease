clear;
seed = 12345678;
rand('seed', seed);
nfolds = 5; nruns=1;
dataname = '';
% load adjacency matrix
y = importdata('associations5.xls');
% y = h5read('../data/disease-circRNA.h5','/g4/lat');
fold_aupr=[];fold_auc=[];fold_accuracy=[];fold_precision=[];fold_recall=[];fold_F1=[];
fold_tpr=[];fold_fpr=[];

%%IsHG是不是超图，1代表是超�?

% globa_true_y_lp=[];
% globa_predict_y_lp=[];
for run=1:nruns
    % split folds
	[num_D,num_G] = size(y);
    for i=1:num_G
% 	crossval_idx = crossvalind('Kfold',y(:),nfolds);
%    
% 
%     for fold=1:nfolds
%         train_idx = find(crossval_idx~=fold);
%         test_idx  = find(crossval_idx==fold);

        y_train = y;
        
		y_train(:,i) = 0;
        
		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
        %% 4. evaluate predictions
        yy=y;

		
% 		test_labels = yy(:,i);
%         [a,b]=size(test_idx);
		predict_scores = RNMFLP_predict(y_train,80,0.001,0.7,4);
        
    
    end
end
% RMSE = sqrt(sum((globa_predict_y_lp-globa_true_y_lp).^2)/length(globa_predict_y_lp))

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

% dlmwrite('data5_denovo_fpr.txt',mean_fpr,'delimiter', '\t');
% dlmwrite('data5_denovo_tpr.txt',mean_tpr,'delimiter', '\t');
% dlmwrite('data5_denovo_recall.txt',mean_recall,'delimiter', '\t');
% dlmwrite('data5_denovo_precision.txt',mean_precision,'delimiter', '\t');
% dlmwrite('data5_denovo_fold_fpr.txt',fold_fpr,'delimiter', '\t');
% dlmwrite('data5_denovo_fold_tpr.txt',fold_tpr,'delimiter', '\t');
% dlmwrite('data5_denovo_fold_recall.txt',fold_recall,'delimiter', '\t');
% dlmwrite('data5_denovo_fold_precision.txt',fold_precision,'delimiter', '\t');
% mean_aupr = mean(fold_aupr)
% mean_auc = mean(fold_auc)
% mean_accurary = mean(fold_accurary)
% mean_precision = mean(fold_precision)
% mean_recall = mean(fold_recall)
% mean_f1 = mean(fold_F1)
% mean_tpr = mean(fold_tpr)
% mean_fpr = mean(fold_fpr)

