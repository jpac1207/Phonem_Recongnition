% Realiza a compactação da entrada 'input', retornando um vetor de
% 'numberOfGroups' colunas atráves das médias do grupos formados por
% 'groupSize' elementos
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