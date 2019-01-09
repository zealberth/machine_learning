function y_hat_all = reje_kmeans_predict(x_test, centros, centros_y, idx_gini, t)
    y_hat_all= [];
    for k = 1:length(t)
        y_hat = [];
        for i=1:size(x_test,1)
            [~, idx] = min(pdist2(x_test(i,:), centros));
            if (idx_gini(idx) < t(k))
                y_hat(i) = centros_y(idx);
            else
                y_hat(i) = 3;
            end
        end
        y_hat_all = [y_hat_all y_hat'];
    end
end