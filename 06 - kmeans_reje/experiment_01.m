% Using k-means to classify with rejection option (only binary problems)

%%
clear; clc; close all; addpath(genpath('utils')); %warning off;

%% experiment setup
n_repetition = 30;
dataset_names = {'irisVir'};      % conjuntos de dados 'column2C', 'bcw' , 'iris_set', 'irisVer', 'irisVir'

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
        [centros, centros_y, idx_gini, t] = reje_kmeans_train(data.x_train, data.y_train);        
		%% test
%         fprintf('Executando classificação...\n');
        y_hat_all = reje_kmeans_predict(data.x_test, centros, centros_y, idx_gini, t);        
		%% get metrics        
        % confusion matrix
        [~,y_test_n] = max(data.y_test,[],2);
        
        for c = 1:size(y_hat_all,2)
            y_hat = y_hat_all(:,c);
            taxa_rejeicao(c) = length(y_hat(y_hat==3))/length(y_hat);
            taxa_erro(c) = mean(y_test_n(y_hat~=3)==y_hat(y_hat~=3));
        end
        
%         cmt(:,:,j) = confusionmat(y_test_n, y_hat); 
        acc(j,:) = taxa_erro;
        reje(j,:) = taxa_rejeicao;
        
        fprintf('Realização de número %d.\n\n', j);        
    end
%     avg_cmt{i} = mean(cmt,3);
    avg_acc{i} = mean(acc);
    svg_reje{i} = mean(reje);
    [mean(acc) mean(reje)];
end


wr = {'0.04', '0.08', '0.12', '0.16', '0.20', '0.24', '0.28', '0.32', '0.36', '0.40', '0.44', '0.48'};
fprintf('Resultados finais(acurácia): \n');
for v=1:size(dataset_names, 2)
%     fprintf('Dataset %s: acuracia %4.2f e rejeicao de %4.2f\n', dataset_names{v}, 100*avg_acc{v}, 100*svg_reje{v});
    figure;
%     axis([0 60 70 100])
%     hold on
    plot(100*svg_reje{v}, 100*avg_acc{v}, '*-')
%     text(100*svg_reje{v}, 100*avg_acc{v}, wr, 'VerticalAlignment','bottom','HorizontalAlignment','right')
    title(sprintf('Dataset %s', dataset_names{v}));
    xlabel('Taxa de acerto (%)')
    ylabel('Taxa de rejeição (%)')
end