% Normaliza os dados da entrada 'X_input' para o domínio [0,1]
function [X_output] =  normalizeInput(X_input)
    X_output = X_input;
    numberOfColumns = size(X_input, 2);
    % Para cada coluna
    for i = 1:numberOfColumns        
        X_max = max(X_output(:, i));
        X_min = min(X_output(:, i));   
        numerator = X_output(:, i) - X_min;
        denominator = (X_max-X_min);
        if denominator ~= 0
            X_output(:, i) = numerator./denominator;
        else
            X_output(:, i) = 0;
        end       
    end        
end