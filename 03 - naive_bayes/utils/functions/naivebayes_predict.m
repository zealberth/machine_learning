function y_hat = naivebayes_predict(x_test, mi, sigma)
    for i=1:size(x_test,1)
        for j=1:size(mi,1)
            temp = exp(((x_test(i,:) - mi(j, :)).^2)./(-2*(sigma(j,:).^2)))./(sqrt(2*pi)*sigma(j,:));
            temp(isnan(temp)) = 1;
            y_hat(i,j) = prod(temp);
        end
    end
    [~, y_hat] = max(y_hat, [], 2);
end