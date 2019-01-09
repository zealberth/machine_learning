function y_hat = window_parzen_predict(x_train, y_train, x_test, h)
    for k=1:size(x_test, 1)
        y_hat(k, :) = zeros(1, size(y_train, 2));
        for i=1:size(y_train, 2)
            a = [];
            [~, y] = max(y_train, [], 2);
            cont = 0;
            for j = 1:size(x_train,1)
                if i == y(j)
                    cont = cont + 1;
                    gaus = exp((-1/(2*h^2)) * ((x_test(k,:) - x_train(j, :)) * (x_test(k,:) - x_train(j, :))')) / (2*pi*h^2)^(1/2);
                    y_hat(k, i) = gaus + y_hat(k, i);
                end
            end
            y_hat(k, i) = y_hat(k, i)/cont;
        end
    end
    [~, y_hat] = max(y_hat, [], 2);
end