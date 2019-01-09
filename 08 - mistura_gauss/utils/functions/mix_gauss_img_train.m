function [mi, sigma, alfa] = mix_gauss_img_train(x_train, k)

    y_train = ones(size(x_train,1),1);
    y_max_idx = max(y_train,[],2);
    j = 1;
    %% Parâmetros iniciais
    
    
%     mi{j}(:,:) = x_train(1:k,:);
    [~, mi{j}(:,:)] = kmeans(x_train, k);
    mi{1}
    for q=1:k
        sigma{j}(:,:,q) = cov(x_train);
        alfa{j}(q) = 1/k;
    end
    

    W_temp = [];
    for q=1:k
        for i=1:size(x_train,1)
            sigma_temp = sigma{j}(:,:,q) + 10^-3 * eye(size(sigma{j}(:,:,q), 1));
            pxw = gaussiana(x_train(i,:), mi{j}(q,:), sigma_temp) * alfa{j}(q);
            px = 0;
            for qq = 1:k
                sigma_temp = sigma{j}(:,:,q) + 10^-3 * eye(size(sigma{j}(:,:,q), 1));
                px = (gaussiana(x_train(i,:), mi{j}(qq,:), sigma_temp) * alfa{j}(qq)) + px;
            end
            W_temp(i,q) = pxw/px;
        end
    end
    W{j} = W_temp;
    
    
    %% Atualização dos parâmetros via E-M
    iteracoes = 10;
        
    for iii = 1:iteracoes        
        [sigma{j}, mi{j}, alfa{j}, W{j}] = EM_algorithm(x_train, sigma{j}, mi{j}, alfa{j}, W{j}, k);  
        iii
    end
    
end
