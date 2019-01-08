% Plot a graph showing accuracy by the number of k neighbors

%%
clear; clc; close all; addpath(genpath('utils')); %warning off;

%% experiment setup
n_repetition = 30;
dataset_names = {'dermatology'};   % datasets 'xor', 'column2C', 'artificial', 'iris', 'column3C'

%% dataset loop
for i=1:length(dataset_names)
    dataset_name = dataset_names{i};
    fprintf('\nExperimento para o dataset %s.\n', dataset_names{i});
    load(sprintf('../dataset/classification/%s.mat', dataset_name))
    params.k = 1:2:size(dataset(1).x_train, 1);
    for l=1:length(params.k)
        %% experiment loop
        for j=1:n_repetition
            %% load/shuffle/divide/normalize dataset
            data = dataset(j);
            %% get params
            k_neighbors = params.k(l);
            %% test
            y_hat = knn_predict(data.x_train, data.y_train, data.x_test, k_neighbors);
            %% get metrics        
            % confusion matrix
            [~,y_test_n] = max(data.y_test,[],2);
            cmt(:,:,j) = confusionmat(y_test_n, y_hat); 
            acc(j) = mean(y_test_n==y_hat);
        end
        avg_acc(l) = mean(acc)*100;
        fprintf('Acurácia de %4.2f para k = %d.\n\n', avg_acc(l), k_neighbors);
    end
    plot(params.k, avg_acc, '-b')
    title(upper(dataset_names{i}))
    xlabel("k vizinhos mais próximos")
    ylabel("acurácia")
    [value, idx] = max(avg_acc);
    fprintf("Melhor parâmetro: k = %d, com acc de %4.2f.", params.k(idx), avg_acc(idx));
end