function [mi, sigma, p_w, t] = reje_bayesmulti_train(x_train, y_train)
    for i=1:size(y_train, 2)
        a = [];
        [~, y] = max(y_train, [], 2);
        for j = 1:size(x_train,1)
            if i == y(j)
                a = [a; x_train(j,:)];
            end
        end
        mi(i,:) = mean(a);
        sigma(:,:,i) = cov(a);
        p_w(i,:) = size(a,1)/size(x_train,1);
    end
    
    wr = 0.04:0.04:0.48;
%     t = 0.03:0.03:0.49;
    t = 0.05:0.05:0.45;
    [~,idx_y_train] = max(y_train,[],2);
    for q = 1:length(wr)
        for k = 1:length(t)
            y_hat = [];
            for i=1:size(y_train,1)
                for j=1:size(mi,1)
                    a = exp(-0.5 * ((x_train(i,:) - mi(j, :)) * inv(sigma(:,:,j)) * (x_train(i,:) - mi(j, :))'));
                    ci =  (2*pi)^(size(mi,1)/2) * det(sigma(:,:,j))^(1/2);
                    p_x(i,j) = (a * p_w(j)) / ci;
                end
            end
            p_x = sum(p_x,2);    
            for i=1:size(y_train,1)
                for j=1:size(mi,1)
                    a = exp(-0.5 * ((x_train(i,:) - mi(j, :)) * inv(sigma(:,:,j)) * (x_train(i,:) - mi(j, :))'));
                    ci = (2*pi)^(size(mi,1)/2) * det(sigma(:,:,j))^(1/2);
                    temp(1, j) = ((a * p_w(j)) / ci) / p_x(i);
                end
                [val, idx] = max(temp,[],2);
                if (val > t(k)+0.5)
                    y_hat(i) = idx;
                else
                    y_hat(i) = 3;
                end
            end
            y_hat = y_hat';
            taxa_rejeicao = length(y_hat(y_hat==3))/length(y_hat);
            taxa_erro = mean(idx_y_train(y_hat~=3)~=y_hat(y_hat~=3));
            matriz_erros(q,k) = taxa_erro + wr(q)*taxa_rejeicao;
        end
    end
    [~, idx] = min(matriz_erros, [], 2);
    t = t(idx);
end