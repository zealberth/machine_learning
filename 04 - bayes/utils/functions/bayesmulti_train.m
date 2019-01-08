function [mi, sigma, p_w] = bayesmulti_train(x_train, y_train)
    for i=1:size(y_train, 2)
        a = [];
        [~, y] = max(y_train, [], 2);
        for j = 1:size(x_train,1)
            if i == y(j)
                a = [a; x_train(j,:)];
            end
        end
        mi(i,:) = mean(a);
        sigma(:,:,i) = cov(a);
        p_w(i,:) = size(a,1)/size(x_train,1);
    end    
end