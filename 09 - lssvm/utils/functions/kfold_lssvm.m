function [avg_acc, avg_std] = kfold_lssvm(x_train, y_train, params)

%% dividindo os k-folds

    indices = crossvalind('Kfold', x_train(:,1), params.kfolds);
    
%% treinamento de cada k-fold
    
     for i = 1:params.kfolds

        dataX_train = x_train(indices~=i,:);
        dataY_train = y_train(indices~=i,:);   
        dataX_test  = x_train(indices==i,:);
        dataY_test  = y_train(indices==i,:);
        
        %% train
        model = lssvm_train(dataX_train, dataY_train, params);
                 
		%% test
        y_hat = lssvm_predict(dataX_test, model);
        
		%% get metrics        
        % accuracy        
        acc(i) = mean(dataY_test==y_hat);
     end
     avg_acc = mean(acc);
     avg_std = std(acc);
end