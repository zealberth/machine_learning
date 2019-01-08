% Cauculate accuracy using Naive Bayes

%%
clear; clc; close all; addpath(genpath('utils')); %warning off;

%% experiment setup
n_repetition = 30;
dataset_names = {'balanceCC'};      % conjuntos de dados 'column2C', 'bcw' , 'iris', 'column3C', 'dermatology' 

%% dataset loop
for i=1:length(dataset_names)
    dataset_name = dataset_names{i};
    fprintf('\nExperimento para o dataset %s.\n', dataset_names{i});
    load(sprintf('../dataset/classification/%s.mat', dataset_name))
	%% experiment loop
    for j=1:n_repetition
		%% load/shuffle/divide/normalize dataset
        data = dataset{j};
        %% train
        [mi, sigma] = naivebayes_train(data.x_train, data.y_train);        
		%% test
        fprintf('Executando classificação...\n');
        y_hat = naivebayes_predict(data.x_test, mi, sigma);        
		%% get metrics        
        % confusion matrix        
        [~,y_test_n] = max(data.y_test,[],2);
        
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
    fprintf('Dataset %s: %4.2f+-%4.2f\n', dataset_names{v}, 100*avg_acc{v}, 100*std_acc{v});
end
