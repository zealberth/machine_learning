function model = dmc_train(x_train, y_train)
    [~, y] = max(y_train,[],2);
    for i=1:size(y_train,2)
        model(i,:) = mean(x_train(y==i, :));
    end
end