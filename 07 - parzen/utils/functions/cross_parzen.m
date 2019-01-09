function params = cross_parzen(x_train, y_train, k)
    fprintf('Executando cross validation...\n');
    %% classifier/regressor params setup
    % parzen parameters h
    h = 0.01:0.03:0.30;  % valor da janela de parzen

    %% experiment loop
    for i=1:length(h)
        %% train
        params = h(i);
        [avg_acc(i), avg_std(i)] = kfold_parzen(x_train, y_train, k, params);
%         fprintf('Janela: %4.2f  |  Acc: %4.2f(+-%4.2f)\n', ...
%             params, 100*avg_acc(i), 100*avg_std(i));
        if avg_acc(i) == 1
            break
        end
    end
    
    %% melhor parametro
    
    [~, idx] = max(avg_acc);
    
    params = h(idx);
    
    fprintf('Parâmetros escolhidos: janela de parzen = %4.2f(Acc = %4.2f+-%4.2f)\n',...
        params, 100*avg_acc(idx), 100*avg_std(idx));
end