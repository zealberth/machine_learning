function y_hat = lssvm_predict(x_test, model)
    K = kernel(model.params, x_test,model.supportVectors);
            
    y_hat = sign(sum(K.*repmat(model.alphas'.*model.supportVectorLabels',size(x_test,1),1),2) + model.bias); %2 + 
%     y_hat(y_hat == 0) = 1;
%     y_hat(y_hat == 3) = 2;
end