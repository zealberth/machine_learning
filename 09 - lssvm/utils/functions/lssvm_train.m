function model = lssvm_train(x_train, y_train, params)
    
    model.supportVectors      = x_train;
    model.supportVectorLabels = y_train;
    model.params = params;
    %% Kernel 
    K = kernel(params, x_train, x_train);

    %% Matriz Omega            
    
    OMEGA = K .* (y_train* y_train') + (1/params.gamma) * eye(size(K));
    
    A = [0 y_train'; y_train OMEGA];
    b = [0; ones(size(OMEGA, 1), 1)];
    
    result = A\b; 
    
    model.bias = result(1,1);
    model.alphas = result(2:end,1);
end