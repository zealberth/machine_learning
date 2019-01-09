function params = cross_lssvm(x_train, y_train, params)
    fprintf('Executando cross validation...\n');
    %% classifier/regressor params setup
    % parzen parameters h
    gamma = [2^-4, 2^-3, 2^-2, 2^-1, 2^0, 2^1, 2^2, 2^3, 2^4];
    sigma = [2^-4, 2^-3, 2^-2, 2^-1, 2^0, 2^1, 2^2, 2^3, 2^4];

    %% experiment loop
    if params.kernel == 1 || params.kernel == 3 % kernel linear
        for i=1:length(gamma)
            params.gamma = gamma(i);
            [avg_acc(i), avg_std(i)] = kfold_lssvm(x_train, y_train, params);
        end
        
    elseif params.kernel == 2 % kernel rbf
        for i=1:length(gamma)
            for j=1:length(sigma)
                params.gamma = gamma(i);
                params.sigma = sigma(j);
                [avg_acc(i,j), avg_std(i,j)] = kfold_lssvm(x_train, y_train, params);
            end
        end
    end
    
    %% melhor parametro
    if params.kernel == 1 || params.kernel == 3
        [~,max_idx] = max(avg_acc);

        params.gamma = gamma(max_idx);

        fprintf('Parâmetros escolhidos: gamma = %4.2f(Acc = %4.2f+-%4.2f)\n',...
            params.gamma, 100*avg_acc(max_idx), 100*avg_std(max_idx));
        
    elseif params.kernel == 2
        [max_num, ~]=max(avg_acc(:));
        [X,Y]=ind2sub(size(avg_acc),find(avg_acc==max_num));
        params.gamma = gamma(X(1));
        params.sigma = sigma(Y(1));

        fprintf('Parâmetros escolhidos: gamma = %4.2f e Abertura = %4.2f(Acc = %4.2f+-%4.2f)\n',...
            params.gamma, params.sigma, 100*avg_acc(X(1),Y(1)), 100*avg_std(X(1),Y(1)));
    end
end