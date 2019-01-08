function y_hat = bayesmulti_predict(x_test, mi, sigma, p_w)
    for i=1:size(x_test,1)
        for j=1:size(mi,1)
            temp = sigma(:,:,j) + 10^-3 * eye(size(sigma(:,:,j), 1));
            a = -0.5 * ((x_test(i,:) - mi(j, :)) * inv(temp) * (x_test(i,:) - mi(j, :))');
            ci = log(p_w(j)) + (size(mi,1)/2) * log(2*pi) + -0.5 *log(det(temp));
            gi(i,j) = a + ci;
        end
    end
    [~, y_hat] = max(gi, [], 2);
end