function [sigma, mi, alfa, W] = EM_algorithm(x_train, sigma, mi, alfa, W, k)
    %% E-Step
    % Calculando pesos
    W_temp = [];
    for q=1:k
        for i=1:size(x_train,1)
            sigma_temp = sigma(:,:,q) + 10^-2 * eye(size(sigma(:,:,q), 1));
            pxw = gaussiana(x_train(i,:), mi(q,:), sigma_temp) * alfa(q);
            px = 0;
            for qq = 1:k
                sigma_temp = sigma(:,:,q) + 10^-2 * eye(size(sigma(:,:,q), 1));
                px = (gaussiana(x_train(i,:), mi(qq,:), sigma_temp) * alfa(qq)) + px;
            end
            W_temp(i,q) = pxw/px;
        end
    end
    W = W_temp;
    
    %% M-Step
    % calculando Nk e atualizando alfas
    Nk = sum(W);
    alfa = Nk ./ size(W,1);

    % atualizando os centros das gaussianas
    for q=1:k
        somador = zeros(size(x_train(1,:)));
        for i=1:size(x_train,1)
            somador = (W(i, q) * x_train(i,:)) + somador;
        end
        mi(q,:) = somador ./ Nk(q);
    end

    % atualizando os matrizes de covariância
    somador = zeros(size(sigma(:,:,1)));
    for q=1:k
        for i=1:size(x_train,1)
            somador = (W(i, q) * ((x_train(i,:) - mi(q,:))' * (x_train(i,:) - mi(q,:)))) + somador;                       
        end
        sigma(:,:,q) = somador ./ Nk(q);
    end
end