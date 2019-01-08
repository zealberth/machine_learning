function dataset = data_load(dataset_name, num_rotulos)
    
    %leitura do dataset
    data = dlmread(sprintf('%s_dataset.csv', dataset_name));
    
    %normalizacao dos dados
    for coluna=1:(size(data,2)-num_rotulos)
        
        data_max = max(data(:, coluna));
        data_min = min(data(:, coluna));
        
        for linha=1:size(data,1)
            data(linha, coluna) = (((data(linha, coluna) - data_min))/...
                (data_max-data_min));
        end
    end
    
    %reordenando de forma aleatoria
    data = data(randperm(end), :);
    
    %divindindo base para treinamento e para teste
    data_train = data(1:floor(0.8*size(data, 1)), :);
    data_test = data(floor(0.8*size(data, 1))+1:end, :);
    
    dataset.x_train = data_train(:, 1:end-num_rotulos);
    dataset.y_train = data_train(:, end-num_rotulos+1:end);
    
    dataset.x_test = data_test(:, 1:end-num_rotulos);
    dataset.y_test = data_test(:, end-num_rotulos+1:end);
end