function y_hat = dmc_predict(model, x_test)
    for i=1:length(x_test)
        [~, idx] = min(pdist2(model, x_test(i, :)));
        y_hat(i,1) = idx;
    end
end