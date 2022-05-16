% Realiza predição da classe de um dado padrão de entrada 'X', utilizando
% os parâmetros: 
% 'hiddenVsInputWeights' -> Matriz que representa os pesos aprendidos para as
% unidades gaussianas
% 'outputVsHiddenWeights' -> Matriz que representa os pesos aprendidos para
% as conexões entre os neurônios escondidos e os neurônios da cama de saída
% 'outputVsHiddenBias' -> Vetor que representa os pesos aprendiso para o
% bias que se conecta a camada de saída
% 'sigmas' -> Raios dos neurônios gaussianos
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

