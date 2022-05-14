phonemesCount = 3;
requiredPatternSize = 30;

test(requiredPatternSize, phonemesCount)

function test(requiredPatternSize, phonemesCount)
    phonemeConfig.phonemClassMap = containers.Map({'DI', 'REI', 'TA', 'ES', 'QUER', 'DA'}, {1, 2, 3, 4, 5, 6}); 
    phonemeConfig.classPhonemMap = containers.Map({1, 2, 3, 4, 5, 6}, {'DI', 'REI', 'TA', 'ES', 'QUER', 'DA'}); 
    phonemeConfig.classFolderMap = containers.Map({1, 2, 3, 4, 5, 6}, {'../Data/Direita/DI/', '../Data/Direita/REI/', '../Data/Direita/TA/', ...
        '../Data/Esquerda/ES/', '../Data/Esquerda/QUER/', '../Data/Esquerda/DA/'});     
    predictions = string();
    for i=1:phonemesCount
        phoneme = upper(input("Escolha um tipo de fonema: ", "s"));
        if(isKey(phonemeConfig.phonemClassMap, phoneme))
            index  = floor(str2double(input("Escolha um índice de áudio (Entre 1 e 40): ", "s")));                      
            if(isnan(index) || (index < 1 || index > 40))
                break
            else
                classId = phonemeConfig.phonemClassMap(phoneme);
                folder = phonemeConfig.classFolderMap(classId);
                audioFile = [folder lower(phoneme) '_' int2str(index) '.wav'];
                [output, prob] = detectPattern(audioFile, requiredPatternSize);                
                predictions(1, i) = phonemeConfig.classPhonemMap(output);
            end 
        else
            break
        end    
    end

    if(i ~= phonemesCount)
        disp('Não foi possível reconhecer a palavra!')
    else
        disp(predictions)
    end
end

% Realiza predição da classe de um dado padrão de entrada 'X', utilizando
% os parâmetros: 
% hiddenVsInputWeights -> Matriz que representa os pesos aprendidos para as
% conexões entre 
function [Y, prob] = testRBF(hiddenVsInputWeights, outputVsHiddenWeights, outputVsHiddenBias, sigmas, X)     
    mi_h = sqrt(sum((X - hiddenVsInputWeights').^2))';            
    Y_h = exp(-((mi_h.^2)./((2*sigmas).^2)));             
    % ------- Output Layer -------    
    net_o = outputVsHiddenWeights * Y_h + outputVsHiddenBias * ones(1, size(Y_h, 2));
    Y_net = exp(net_o)/sum(exp(net_o));  % Aplicação da softmax                               
    [~, index] = max(Y_net);
    Y = index;
    prob = Y_net(index);
end

% Realiza o carregamento dos pesos, aplica normalização e invoca função de
% execução da RBF
function[prediction, prob] = predictExampleUsingBestWeights(inputPattern)
    weightsStruct = load('bestWeights.mat');
    hiddenVsInputWeights = weightsStruct.hiddenVsInputWeights;    
    outputVsHiddenWeights = weightsStruct.outputVsHiddenWeights;
    outputVsHiddenBias = weightsStruct.outputVsHiddenBias;
    sigmas = weightsStruct.sigmas;   
    processed_dataset = load('processed_dataset.mat');
    X = processed_dataset.X;    
    X = [X ; inputPattern];           
    X_norm = normalizeInput(X);    
    [prediction, prob] = testRBF(hiddenVsInputWeights, outputVsHiddenWeights, outputVsHiddenBias, sigmas, X_norm(end, :)');      
end

% Recebe o caminho do arquivo de audio via atributo 'audioPath', além do
% tamanho padronizado pela entrada via 'requiredPatternSize'
function[output, prob] =  detectPattern(audioPath, requiredPatternSize)
    samples = audioread(audioPath);
    fftResult = abs(fft(samples));
    fftHalf = floor(size(fftResult, 1)/2);   
    fftResult = fftResult(1:fftHalf, 1);
    groupSize = floor(size(fftResult, 1)/requiredPatternSize);   
    inputPattern = compactInput(fftResult, requiredPatternSize, groupSize);   
    [output, prob] = predictExampleUsingBestWeights(inputPattern);
end