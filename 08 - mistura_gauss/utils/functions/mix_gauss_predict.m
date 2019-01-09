function y_hat = mix_gauss_predict(x_test, mi, sigma, alfa, k, p_w)
    for i=1:size(x_test,1)
            for j=1:length(mi)
                for q=1:k
                    sigma_temp = sigma{j}(:,:,q) + 10^-3 * eye(size(sigma{j}(:,:,q), 1));
                    p_x_temp(q) = gaussiana(x_test(i,:), mi{j}(q,:), sigma_temp) * alfa{j}(q); 
                end
                gi(i,j) = sum(p_x_temp) * p_w(j);
            end
        end
    [~, y_hat] = max(gi, [], 2);
end