close all; clear; clc;

for i=1:30
   dataset(i) = data_load('csvs/sin_set',1);
end
data =dataset(1);
plot(data.x_train, data.y_train,'.')
