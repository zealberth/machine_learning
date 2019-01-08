function y_hat = knn_predict(x_train, y_train, x_test, k_neighbors)   
    for i=1:length(x_test)
        [~, y] = max(y_train,[],2);
        [~, idx] = sort(pdist2(x_train, x_test(i,:)));
        y = y(idx);
        y_hat(i,1) = mode(y(1:k_neighbors));
    end    
end