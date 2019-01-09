% Least Squares Support Vector Machines to classification

%%
clear; clc; close all; addpath(genpath('utils')); %warning off;

%% experiment setup
n_repetition = 30;
dataset_names = {'column2C'};      % conjuntos de dados 'column2C', 'bcw' , 'irisSet', 'irisVer', 'irisVir'
params.kfolds = 10;
params.kernel = 2; % 1 - linear / 2 - rbf gaussian


%% dataset loop
for i=1:length(dataset_names)
    dataset_name = dataset_names{i};
    fprintf('\nExperimento para o dataset %s.\n', dataset_names{i});
    load(sprintf('../dataset/classification/%s.mat', dataset_name))
	%% experiment loop
    for j=1:n_repetition
		%% load/shuffle/divide/normalize dataset
        data = dataset(j);   
        %% train
        [~,y_train_n] = max(data.y_train,[],2);
        y_train_n(y_train_n==2) = -1;
        
        params = cross_lssvm(data.x_train, y_train_n, params);
        model = lssvm_train(data.x_train, y_train_n, params);
		%% test
        y_hat = lssvm_predict(data.x_test, model);   
		%% get metrics        
        % confusion matrix
        [~,y_test_n] = max(data.y_test,[],2);
        y_test_n(y_test_n==2) = -1;
        
        cmt(:,:,j) = confusionmat(y_test_n, y_hat); 
        acc(j) = mean(y_test_n==y_hat);
        
        fprintf('Taxa de acerto de %4.2f para a realização de número %d.\n\n', 100*acc(j), j);        
    end
    avg_cmt{i} = mean(cmt,3);
    avg_acc{i} = mean(acc);
    std_acc{i} = std(acc);
    [mean(acc) std(acc)];
end

fprintf('Resultados finais(acurácia): \n');
for v=1:size(avg_acc, 2)
    test(v,:) = [avg_acc{v} std_acc{v}];
    fprintf('\n');
    fprintf('Dataset %s: %4.2f+-%4.2f\n', dataset_names{v}, 100*avg_acc{v}, 100*std_acc{v});
    T=strrep(evalc('disp(avg_cmt{v})'),'.',',');
    fprintf(T);
end
