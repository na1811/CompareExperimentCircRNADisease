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

fold_aupr=[];fold_auc=[];fold_accuracy=[];fold_precision=[];fold_recall=[];fold_F1=[];
fold_tpr=[];fold_fpr=[];

% globa_true_y_lp=[];
% globa_predict_y_lp=[];
for run=1:nruns
    % split folds
	[num_D,num_G] = size(y);
    for i=1:num_G

        y_train = y;
		y_train(:,i) = 0;
        %%4. evaluate predictions
        yy=y;
		predict_scores = RNMFLP_predict(y_train,80,0.001,0.7,4);       
    
    end
end


