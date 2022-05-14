% ---------- Parâmetros Gerais ----------
maxEpochs = 50000; % Número de épocas do treinamento
numberOfTrainings = 10; % Número de treinamentos a serem utilizados
H = 30; % Número de neurônios na camada escondida
I = 30; % Número de neurônios na camada de entrada
O = 6; % Número de neurônios na camada de saída
eta = 0.05; % Learning Rate utilizado no cálculo do backpropagation.
eta_gaussian = 0.1; % Learning Rate utilizado no cálculo da atualização de centro dos neurônios de ativação gaussiana.

% Realiza treinamento da RBF 'numberOfTrainings' vezes.
doTraining(maxEpochs, numberOfTrainings, I, H, O, eta, eta_gaussian);

% Realiza 'numberOfTrainings' treinamentos, obtendo ao final:
% Melhor erro de treinamento encontrado
% Média dos erros de treinamento
% Média dos erros de validação
% Gráfico com os erros médios por epóca
function doTraining(maxEpochs, numberOfTrainings, I, H, O, eta, eta_gaussian)        
    processed_dataset = load('processed_dataset.mat');
    X = processed_dataset.X;
    Y = processed_dataset.Y;
    X_norm = normalizeInput(X);    
    [X_train, Y_train, X_val, Y_val, X_test, Y_test] = splitData(X_norm, Y);
    finalErrors = zeros(maxEpochs, 1);  
    finalValErrors = zeros(maxEpochs, 1);
    bestError = 1;     
    
    for i = 1:numberOfTrainings
        [hiddenVsInputWeights, outputVsHiddenWeights, outputVsHiddenBias, sigmas, errors, valErrors]  = trainRBF(I, H, O, maxEpochs, eta, eta_gaussian, ...
            X_train', Y_train, X_val', Y_val); 
        finalErrors = finalErrors + errors;
        finalValErrors = finalValErrors + valErrors;
        if(errors(maxEpochs) < bestError)
            bestError = errors(maxEpochs);
            save('bestWeights.mat', 'hiddenVsInputWeights', 'outputVsHiddenWeights', 'outputVsHiddenBias', 'sigmas');
        end        
    end
    meanFinalErrors = (finalErrors./numberOfTrainings);
    meanFinalValErrors = (finalValErrors./numberOfTrainings);
    bestError
    meanFinalError = meanFinalErrors(maxEpochs)
    meanFinalValError = meanFinalValErrors(maxEpochs)
    plot((1:maxEpochs), meanFinalErrors, 'o');
    hold on;
    plot((1:maxEpochs), meanFinalValErrors, 'x');
    hold off;
    legend('Média Erros Treinamento', 'Média Erros Validação');
    ylabel('Erro Quadrático Médio');
    xlabel('Épocas');
    title('Erros de Treino e Validação do Treinamento');
end

% Realiza o treinamento da RBF, de acordo com os parametros:
% I -> Número de neurônios na camada de entrada
% H -> Número de neurônios na camada escondida
% O -> Número de neurônios na camada de saída
% maxEpochs -> Número de epócas do treinamento
% eta -> Taxa de aprendizado
% activationType -> Flag utilizada para definir a função de ativação da
% camada escondida
% X_train -> Padrões de entrada utilizados durante o treinamento
% Y_train -> Padrões de saída utilizados durante o treinamento
% X_val -> Padrões de entrada utilizados na validação
% Y_val -> Padrões de saída utilizados na validação
function [hiddenVsInputWeights, outputVsHiddenWeights, outputVsHiddenBias, sigmas, finalErrors, finalValErrors] = trainRBF(I, H, O, maxEpochs, eta, ...
    eta_gaussian, X_train, Y_train, X_val, Y_val)
    currentEpoch = 1;    
    errors = zeros(maxEpochs, 1);  
    validationErrors = zeros(maxEpochs, 1);
    % Número de padrões de entrada
    numberOfTrainingInstances = size(X_train, 2);
    % Número de padrões de validação
    numberOfValidationInstances = size(X_val, 2);    
    % Centros camada escondida
    C = rand(H, I) - 0.5;     
    % Pesos entre camada escondida e camada de saída
    Woh = rand (O, H) - 0.5;
    % Bias entre camada escondida e camada de saída
    bias_oh = rand(O, 1) - 0.5;    

    % ---------------------- Aplicação do Algoritmo WTA ----------------------    
    C = wta(X_train, C, eta_gaussian);

    % ---------------------- Determinação da abertura dos neurônios escondidos ----------------------

    % Considera os N/2 neurônios mais próximos para cálculo da abertura de
    % cada neurônio
    T = floor(H/2);
    % Vetor que irá armazenar a abertura para cada neurônio da camada escôndida
    sigmas = zeros(H, 1);
    % Percorre todos os neurônios da camada escondida
    distancesBetweenHiddenNeurons = zeros(H, H) + realmax;    
    % Computa a distância entre cada par de neurônios 
    for i=1:H
        for j=i+1:H            
            distanceBetweenNeuronsIandJ = sqrt(sum((C(i, :) - C(j, :)).^2));            
            distancesBetweenHiddenNeurons(i, j) = distanceBetweenNeuronsIandJ;
            distancesBetweenHiddenNeurons(j, i) = distanceBetweenNeuronsIandJ;
        end
    end
    
    % Computa a abertura de cada neurônio escondido     
    % Percorre todos os neurônios da camada escondida
    for i=1:H
        % Vetor que irá armazenar as T menores distâncias do neurônio i em
        % relação aos outros neurônios escondidos
        minDistances = zeros(T, 1);        
        for j=1:T                        
            [minValue, minPosition] = min(distancesBetweenHiddenNeurons(i, :));
            minDistances(j) = minValue;
            distancesBetweenHiddenNeurons(i, minPosition) = realmax;            
        end
        sigmas(i) = sum(minDistances)/T;
    end    
     
    % ---------------------- Treinamento da camada de saída ----------------------    
    while currentEpoch <= maxEpochs    
        trainingPredictions = zeros(O, numberOfTrainingInstances);
        validationPredictions = zeros(O, numberOfValidationInstances);
        for i=1:numberOfTrainingInstances          
             % ------- Hidden Layer -------            
             mi_h = sqrt(sum((X_train(:, i) - C').^2))'; %OK             
             Y_h = exp(-((mi_h.^2)./((2*sigmas).^2)));             
             % ------- Output Layer -------    
             net_o = Woh * Y_h + bias_oh * ones(1, size(Y_h, 2));
             Y_net = exp(net_o)/sum(exp(net_o));  % Aplicação da softmax                          
             E = (-1).*sum((Y_train(:, i).*log(Y_net)));  % Computação do erro                
             trainingPredictions(:, i) = Y_net;
             % backward                 
             df =  (Y_train(:, i)-Y_net);             
             delta_bias_oh = eta * sum((E.*df)')';             
             delta_Woh = eta * (E.*df)*Y_h';       
             
             % update weights  
             Woh = Woh + delta_Woh;
             bias_oh = bias_oh + delta_bias_oh;            
           
        end                       
        error = sum(((Y_train .* (1-trainingPredictions)).^2), 'all')/numberOfTrainingInstances;
        %sprintf("%f", error)
        errors(currentEpoch) = error;
        
        % Validação
        for i=1:numberOfValidationInstances 
              % ------- Hidden Layer -------            
             mi_h = sqrt(sum((X_val(:, i) - C').^2))'; %OK             
             Y_h = exp(-((mi_h.^2)./((2*sigmas).^2)));             
             % ------- Output Layer -------    
             net_o = Woh * Y_h + bias_oh * ones(1, size(Y_h, 2));
             Y_net = exp(net_o)/sum(exp(net_o));  % Aplicação da softmax                                                 
             validationPredictions(:, i) = Y_net;
        end
        validationError = sum(((Y_val .* (1-validationPredictions)).^2), 'all')/numberOfValidationInstances;         
        %sprintf("%f", validationError);
        validationErrors(currentEpoch) = validationError;

        currentEpoch = currentEpoch + 1;
   end     

    finalErrors = errors;
    finalValErrors = validationErrors;
    hiddenVsInputWeights = C;   
    outputVsHiddenWeights = Woh;
    outputVsHiddenBias = bias_oh;
    sigmas = sigmas;
end

% Retorna a posição na matriz 'hiddenNeurons' cujo elemento é mais próximo
% do vetor coluna 'inputPattern'
function[minPosition] = getNearestNeuronPosition(inputPattern, hiddenNeurons)
    differences = zeros(size(hiddenNeurons, 1), 1);    
    for j = 1:size(hiddenNeurons, 1)           
        absoluteDifference  = sum((inputPattern - hiddenNeurons(j, :)').^2);
        differences(j) = absoluteDifference;
    end       
    [~, minPosition] = min(differences);     
end

% Aplicação do algoritmo WTA. Recebe como argumento, a matrix 'inputMatrix'
% com os padrões de entrada, a matrix 'hiddenNeurons' com os pesos dos
% neurônios escondidos e o eta relativo a atualização dos centros dos neurônios escondidos. Como retorno, são devolvidos:
% 'nearestHiddenNeurons' -> Vetor coluna contendo a posição do neurônio
% mais próximo para cada padrão de entrada;
% 'hiddenNeurons' -> Neurônios escondidos com os valores de centro
% atualizados
function [hiddenNeurons] = wta(inputMatrix, hiddenNeurons, eta_gaussian)    
    previousQuantizationError = realmax;
    howManyIterations = 0;
    maxOfIterations = 100;
    numberOfInstances = size(inputMatrix, 2);    
    while true       
        quantizationError = 0;
        % Percorre todos os vetores de entrada x
        for i = 1:numberOfInstances                 
           % Para cada vetor de entrada, determina o centro mais próximo                      
           minPosition = getNearestNeuronPosition(inputMatrix(:, i), hiddenNeurons);          
           % Atualiza o centro mais próximo             
           hiddenNeurons(minPosition, :) = hiddenNeurons(minPosition, :) + (eta_gaussian * (inputMatrix(:, i)' - hiddenNeurons(minPosition, :)));                 
           % Computa erro de quantização
           quantizationError = quantizationError + sum(sqrt(((inputMatrix(:, i)' - hiddenNeurons(minPosition, :)).^2)).^2);
        end       
        quantizationError = quantizationError/numberOfInstances;
        howManyIterations = howManyIterations + 1;        
        % Condições de Parada: maxOfIterations ou erro não diminuiu desde a
        % última iteração          
        %howManyIterations
        %sprintf("%f", quantizationError)
        %sprintf("%f", previousQuantizationError)
        if(quantizationError < previousQuantizationError && howManyIterations <= maxOfIterations)
            previousQuantizationError = quantizationError;
        else            
            break;
        end    
    end  
end

% Realiza a divisão dos dados contidos em 'X' e 'Y' em:
% X_train -> Padrões de entrada a serem utilizados no treino (70%)
% Y_train -> Padrões de saída a serem utilizados no treino (70%)
% X_val -> Padrões de entrada a serem utilizados na validação (20%)
% Y_val -> Padrões de saída a serem utilizados na validação (20%)
% X_test -> Padrões de entrada a serem utilizados no teste (10%)
% Y_test -> Padrões de saída a serem utilizados no testw (10%)
function [X_train, Y_train, X_val, Y_val, X_test, Y_test] = splitData(X, Y)
    numberOfRows = size(X, 1);
    trainProportion = 0.65;
    trainRows = floor(numberOfRows * trainProportion);
    valProportion = 0.25;
    valRows = floor(numberOfRows * valProportion);
    testProportion = 0.1;
    testRows = floor(numberOfRows * testProportion);    

    randIndexes = randperm(numberOfRows);   
    trainIndexes = randIndexes(1:trainRows);    
    initOfValRows = (trainRows + 1);
    valIndexes = randIndexes(initOfValRows:(initOfValRows + valRows - 1));
    initOfTestRows = (initOfValRows + valRows);
    testIndexes = randIndexes(initOfTestRows:(initOfTestRows + testRows - 1));

    X_train = X(trainIndexes, :);
    Y_train = Y(:, trainIndexes);
    
    X_val = X(valIndexes, :);
    Y_val = Y(:, valIndexes);
    
    X_test = X(testIndexes, :);
    Y_test = Y(:, testIndexes);
end