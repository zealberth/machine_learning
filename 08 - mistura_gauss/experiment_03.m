% Segmentation of images using GMM

%%
clear; clc; close all; addpath(genpath('utils')); %warning off;

%% experiment setup
n_repetition = 1;

%% experiment loop
for j=1:n_repetition
    %% load/shuffle/divide/normalize dataset
    data = imread('images/usa.png'); % 'usa.png', 'brasil.jpg'
    I2 = im2double(data);
    r = I2(:,:,1);
    g = I2(:,:,2);
    b = I2(:,:,3);
    rgb= [r(:) g(:) b(:)];
    %% train
    k = 3;
    [mi, sigma, alfa] = mix_gauss_img_train(rgb, k);        
    %% test
    fprintf('Executando classificação...\n');
    y_hat = mix_gauss_img_predict(rgb, mi, sigma, k);        
    %% get metrics        
    % confusion matrix
    for q=1:k
        temp = rgb;
        temp(y_hat==q,:) = temp(y_hat==q,:)*0;
        R = reshape(temp(:,1),size(r));
        G = reshape(temp(:,2),size(r));
        B = reshape(temp(:,3),size(r));
        image = cat(3, R,G,B);
        figure, imshow(image)       
    end
end

