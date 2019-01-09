function y_hat = mix_gauss_img_predict(x_test, mi, sigma, k)
    for i=1:size(x_test,1)
            for j=1:length(mi)
                for q=1:k
                    sigma_temp = sigma{j}(:,:,q) + 10^-3 * eye(size(sigma{j}(:,:,q), 1));
                    p_x_temp(i, q) = gaussiana(x_test(i,:), mi{j}(q,:), sigma_temp); 
                end
            end
        end
    [~, y_hat] = max(p_x_temp, [], 2);
end