function [centros, centros_y, idx_gini, t] = reje_kmeans_train(x_train, y_train)
    k = 10;
    
    [idx_centros, centros] = kmeans(x_train,k);
    [~,y_train_n] = max(y_train, [], 2);
    
    centros_y = [];
    for i=1:k
        rotulo = mode(y_train_n(idx_centros==i));
        centros_y = [centros_y rotulo];
    end
    
    idx_gini = [];
    for i=1:k
        temp = y_train_n(idx_centros==i);
        val = 1 - ((length(temp(temp==1))/length(temp))^2 + (length(temp(temp==2))/length(temp))^2);
        idx_gini = [idx_gini val];
    end
    
    wr = 0.04:0.04:0.48;
    t = 0.03:0.03:0.49;
    for q = 1:length(wr)
        for k = 1:length(t)
            y_hat = [];
            for i=1:size(y_train,1)
                if (idx_gini(idx_centros(i)) < t(k))
                    y_hat(i) = centros_y(idx_centros(i));
                else
                    y_hat(i) = 3;
                end
            end
            y_hat = y_hat';
            taxa_rejeicao = length(y_hat(y_hat==3))/length(y_hat);
            taxa_erro = mean(y_train_n(y_hat~=3)~=y_hat(y_hat~=3));
            matriz_erros(q,k) = taxa_erro + wr(q)*taxa_rejeicao;
        end
    end
    [~, idx] = min(matriz_erros, [], 2);
    t = t(idx);
    
end