function [mi, sigma, alfa, p_w] = mix_gauss_train(x_train, y_train, k)
    [~, y_max_idx] = max(y_train, [], 2);
    %% Parâmetros iniciais   
    for j=1:size(y_train, 2)
        temp = x_train(y_max_idx==j,:);
        mi{j}(:,:) = temp(1:k,:);
        for q=1:k
            sigma{j}(:,:,q) = cov(x_train);
            alfa{j}(q) = 1/k;
        end
        p_w(j) = sum(y_max_idx==j)/size(x_train, 1);
    end
    
    for j=1:size(y_train,2)
        temp = x_train(y_max_idx==j,:);
        W_temp = [];
        for q=1:k
            for i=1:size(temp,1)
                sigma_temp = sigma{j}(:,:,q) + 10^-3 * eye(size(sigma{j}(:,:,q), 1));
                pxw = gaussiana(temp(i,:), mi{j}(q,:), sigma_temp) * alfa{j}(q);
                px = 0;
                for qq = 1:k
                    sigma_temp = sigma{j}(:,:,q) + 10^-3 * eye(size(sigma{j}(:,:,q), 1));
                    px = (gaussiana(temp(i,:), mi{j}(qq,:), sigma_temp) * alfa{j}(qq)) + px;
                end
                W_temp(i,q) = pxw/px;
            end
        end
        W{j} = W_temp;
    end
    
    %% Atualização dos parâmetros via E-M
    iteracoes = 20;
    
    classe_plot = 3;
    
%     if (size(x_train, 2) == 2) % plotar apenas se for no R²
% %         figure
% %         hold on
%         for classe=1:length(unique(y_max_idx))
%             figure
%             hold on
%             classe_plot = classe;
%             plot(x_train(y_max_idx == classe_plot, 1),x_train(y_max_idx == classe_plot, 2),'b*');
%             plot(x_train(y_max_idx ~= classe_plot, 1),x_train(y_max_idx ~= classe_plot, 2),'r*');
% %             plot(x_train(:, 1),x_train(:, 2),'r*');
%             for q=1:k
%                 mu = mi{classe_plot}(q,:); %// data
%                 sigmaaa = sigma{classe_plot}(:,:,q); %// data
%                 x = 0:.01:1; %// x axis
%                 y = 0:.01:1; %// y axis
% 
%                 [X Y] = meshgrid(x,y); %// all combinations of x, y
%                 Z = mvnpdf([X(:) Y(:)],mu,sigmaaa); %// compute Gaussian pdf
%                 Z = reshape(Z,size(X)); %// put into same size as X, Y
% 
%                 contour(X,Y,Z), axis equal  %// contour plot; set same scale for x and y...
%             end
%         end
%         hold off;
%     end
    
    for iii = 1:iteracoes
        for j=1:size(y_train, 2)
            temp = x_train(y_max_idx==j,:);
            [sigma{j}, mi{j}, alfa{j}, W{j}] = EM_algorithm(temp, sigma{j}, mi{j}, alfa{j}, W{j}, k);
        end
    end
    
%     if (size(x_train, 2) == 2) % plotar apenas se for no R²
% %         figure
% %         hold on
%         for classe=1:length(unique(y_max_idx))
%             figure
%             hold on
%             classe_plot = classe;
%             plot(x_train(y_max_idx == classe_plot, 1),x_train(y_max_idx == classe_plot, 2),'b*');
%             plot(x_train(y_max_idx ~= classe_plot, 1),x_train(y_max_idx ~= classe_plot, 2),'r*');
% %             plot(x_train(:, 1),x_train(:, 2),'r*');
%             for q=1:k
%                 mu = mi{classe_plot}(q,:); %// data
%                 sigmaaa = sigma{classe_plot}(:,:,q); %// data
%                 x = 0:.01:1; %// x axis
%                 y = 0:.01:1; %// y axis
% 
%                 [X Y] = meshgrid(x,y); %// all combinations of x, y
%                 Z = mvnpdf([X(:) Y(:)],mu,sigmaaa); %// compute Gaussian pdf
%                 Z = reshape(Z,size(X)); %// put into same size as X, Y
% 
%                 contour(X,Y,Z), axis equal  %// contour plot; set same scale for x and y...
%             end
%         end
%         hold off;
% %         close all;
%     end
end
