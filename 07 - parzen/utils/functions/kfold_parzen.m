function [avg_acc, avg_std] = kfold_parzen(x_train, y_train, k, h)

%% dividindo os k-folds

    indices = crossvalind('Kfold', x_train(:,1), k);
    
%% treinamento de cada k-fold
    
     for i = 1:k

        dataX_train = x_train(indices~=i,:);
        dataY_train = y_train(indices~=i,:);   
        dataX_test  = x_train(indices==i,:);
        dataY_test  = y_train(indices==i,:);
                 
		%% test
        y_hat = window_parzen_predict(dataX_train,dataY_train, dataX_test, h);
        
		%% get metrics        
        % accuracy
        [~,y_test_n] = max(dataY_test,[],2);
        
        acc(i) = mean(y_test_n==y_hat);
     end
     avg_acc = mean(acc);
     avg_std = std(acc);
end