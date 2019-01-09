function y_hat_all = reje_bayesmulti_predict(x_test, mi, sigma, p_w, t)
    y_hat_all= [];
    for q = 1:length(t)
        y_hat = [];
        for i=1:size(x_test,1)
            for j=1:size(mi,1)
                a = exp(-0.5 * ((x_test(i,:) - mi(j, :)) * inv(sigma(:,:,j)) * (x_test(i,:) - mi(j, :))'));
                ci = 1/( (2*pi)^(size(mi,1)/2) * det(sigma(:,:,j))^(1/2));
                p_x(i,j) = (a * ci)* p_w(j);
            end
        end
        p_x = sum(p_x,2);    
        for i=1:size(x_test,1)
            for j=1:size(mi,1)
                a = exp(-0.5 * ((x_test(i,:) - mi(j, :)) * inv(sigma(:,:,j)) * (x_test(i,:) - mi(j, :))'));
                ci = 1/( (2*pi)^(size(mi,1)/2) * det(sigma(:,:,j))^(1/2));
                temp(1, j) = ((a * ci)* p_w(j)) / p_x(i);
            end
            [val, idx] = max(temp,[],2);
            if (val > t(q)+0.5)
                y_hat(i) = idx;
            else
                y_hat(i) = 3;
            end
        end
        y_hat_all = [y_hat_all y_hat'];
    end
end