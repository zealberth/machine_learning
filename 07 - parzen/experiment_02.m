clear; clc; close all; addpath(genpath('utils'));

%% experiment setup
n_repetition  = 1;              % quantidade de realizações
dataset_names = {'irisSet'};       % conjuntos de dados 'irisSet', 'irisVer', 'irisVir'


%% dataset loop
for i=1:length(dataset_names)
    dataset_name = dataset_names{i};
    load(sprintf('../dataset/%s.mat', dataset_name))
	%% experiment loop
    for j=1:n_repetition
		%% load/shuffle/divide/normalize dataset
        data = dataset(i);
        combinations = combnk(1:size(data.x_train,2),2);
        for k=1:size(combinations,1)
            % get patterns with two attributes
            x_train = data.x_train(:,combinations(k,:)); 
            %% train
%             [centros, centros_y, idx_gini, t] = reje_kmeans_train(x_train, data.y_train);
            h = cross_parzen(x_train, data.y_train, 10);
            %% plot decision surface
            x_min = min(x_train);
            x_max = max(x_train);
            
            
            [x, y] = meshgrid(linspace(x_min(1), x_max(1)), linspace(x_min(2),x_max(2)));
            image_size = size(x);
            xy = [x(:) y(:)];
            
%             y_hat = reje_kmeans_predict(xy, centros, centros_y, idx_gini, t);
            y_hat = window_parzen_predict(x_train, data.y_train, xy, h);
            
            decisionmap = reshape(y_hat, image_size);
            figure,
            img = imagesc([x_min(1) x_max(1)],[x_min(2) x_max(2)],decisionmap);
            hold on;
            set(gca,'ydir','normal');
            cmap = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1];%clc
            colormap(cmap);
            [~,y_test_n] = max(data.y_train,[],2);
            plot(x_train(y_test_n == 1, 1),x_train(y_test_n == 1, 2),'r*');
            plot(x_train(y_test_n == 2, 1),x_train(y_test_n == 2, 2),'g*');
            plot(x_train(y_test_n == 3, 1),x_train(y_test_n == 3, 2),'b*');
            legend({'class 1', 'class 2', 'class 3'});
            title(upper(dataset_names{i}));
            
            xlabel(sprintf('feature %d', combinations(k,1)));
            ylabel(sprintf('feature %d', combinations(k,2)));
            
            hold off;
        end
    end
end