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

for run=1:nruns
    % split folds
	[num_D,num_G] = size(y)
	crossval_idx = crossvalind('Kfold',y(:),nfolds);
   
    for fold=1:nfolds
        train_idx = find(crossval_idx~=fold);
        test_idx  = find(crossval_idx==fold);

        y_train = y;
        		
        %% 4. evaluate predictions
        yy=y;

		test_labels = yy(test_idx);
		predict_scores = RNMFLP_predict(y_train,80,0.001,0.7,4);    

        var_name = sprintf('RNMFLP%d', fold);
        assignin('base', var_name, predict_scores);
    end
end
