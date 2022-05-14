% Este arquivo é executado de modo a transformar os arquivos de áudio em um
% dataset estruturado, aplicando compactação por média.
requiredPatternSize = 30; % Quantidade de atributos requeridas para cada padrão de entrada
phonemesCount = 6; % Quantidade de fonemas/classes
patternsByPhonem = 40; % Quantidade de padrões para cada fonema
plotInstances = 1; % Define se a intâncias serão exibidas em forma gráfica após o tratamento.

[X, Y] = loadData(requiredPatternSize, (phonemesCount*patternsByPhonem), phonemesCount, plotInstances);
save("processed_dataset.mat", "X", "Y");

function[Y] = applyHotEncode(optionsCount, label)
    Y = zeros(optionsCount, 1);
    for j = 1:optionsCount                
        if(j == label)
            Y(j, 1) = 1;
        else
            Y(j, 1) = 0;
        end
    end       
end

function[X, Y] = loadData(requiredPatternSize, totalInstances, phonemesCount, plotInstances)
    phonemeConfig.classMap = containers.Map({'../Data/Direita/DI/', '../Data/Direita/REI/', '../Data/Direita/TA/', ...
        '../Data/Esquerda/ES/', '../Data/Esquerda/QUER/', '../Data/Esquerda/DA/'}, {1, 2, 3, 4, 5, 6}); 
    X = zeros(totalInstances, requiredPatternSize);
    Y = zeros(phonemesCount, totalInstances);
    instanceCount = 1;
    % Itera sobre todas as pastas
    for key = keys(phonemeConfig.classMap)
        %key{1}
        filesInDirectory = dir(key{1});
        filesInDirectorySize = size(filesInDirectory, 1); 
        % Itera sobre todos os arquivos na pasta atual
        for i=1:filesInDirectorySize
            file = filesInDirectory(i);
            % Se o elemento atual for um arquivo
            if ~file.isdir
                fileCompletePath = strcat(file.folder, '\', file.name);
                samples = audioread(fileCompletePath);
                fftResult = abs(fft(samples));
                fftHalf = floor(size(fftResult, 1)/2);                 
                fftResult = fftResult(1:fftHalf, 1); % Utiliza apenas metade dos valores                  
                groupSize = floor(size(fftResult, 1)/requiredPatternSize);                
                inputPattern = compactInput(fftResult, requiredPatternSize, groupSize); 
                if plotInstances 
                    plot(inputPattern);
                    hold on;                
                end
                X(instanceCount, :) = inputPattern(1, :);               
                % Creating Y
                label =  phonemeConfig.classMap(key{1});
                Y(:, instanceCount) = applyHotEncode(phonemesCount, label);
                instanceCount = instanceCount + 1;
            end
        end
        if plotInstances 
            hold off;
            phonemName = split(key, '/');       
            title(['Padrões de Entrada Fonema: '  phonemName{4, 1}]);
            xlabel('Grupo');
            ylabel('Média das Amplitudes do Grupo');
            saveas(gcf, ['training_figures/' 'Padrões_de_Entrada_Fonema_'  phonemName{4, 1} '.png'])
        end
    end  
    %Y
    %max
end

% Realiza a compactação da entrada 
function[X_output] = compactInput(input, numberOfGroups, groupSize)    
    X_output = zeros(1, numberOfGroups);      
    startPosition = 1;
    % Para cada atributo no padrão de saída, cálcula a média do padrão
    % de entrada
    for j=1:numberOfGroups        
        slice = input(startPosition:(startPosition + groupSize - 1));
        X_output(1, j) = mean(slice);
        startPosition = startPosition + groupSize;
    end  
end