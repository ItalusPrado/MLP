clc;
clear all;
close all;

%Carregando dados
data = load('iris.txt');

%Iniciando alguns valores padrões
maxPeriods = 10000;
alpha = 0.01;

%Separando entre treino e test
data = data(randperm(size(data,1)),:);
trainSize = round(size(data,1)/10*8);
train = data(1:trainSize,:);
test = data(trainSize+1:size(data,1),:);

samples = trainSize;
inputs = size(data,2)-2; % -3 Dos valores esperados, +1 do bias

bias = repmat(-1,samples,1);

%Adicionando o bias e normalizando
train(:,1:4) = (train(:,1:4) - repmat(min(train(:,1:4)),120,1))./repmat((max(train(:,1:4))-min(train(:,1:4))),120,1);
test(:,1:4) = (test(:,1:4) - repmat(min(test(:,1:4)),30,1))./repmat((max(test(:,1:4))-min(test(:,1:4))),30,1);
train = [bias train];
test = [bias(1:30) test];

periodError = 0;
period = 0;
Y = zeros(samples,3);

%Separando os valores de entradas X dos valores desejados D
D = train(:,inputs+1:size(train,2));
X = train(:,1:inputs);
    

hiddenNeurons = size(X',2)/30;
samples          = size(X,1);
inputs         = size(X,2); 

% Vetor para armazenar os erros de cada época.
MSE = zeros(1,maxPeriods); 

convergence = 0;          % Variável que atesta a convergência (0: FALSO)

while (convergence == 0)
    
    % Geração dos pesos: há este laço mais externo para caso não haja con-
    % vergência, o processo seja reiniciado com novos pesos.
    
    wHidden = (rand(inputs,hiddenNeurons) - 0.5)/10;   
    wExit  = (rand(3,hiddenNeurons) - 0.5)/10;
    
    for period = 1:maxPeriods
    
        % Laço para a varredura de todos os padrões.
        for j = 1:samples

            x = X(j,:); % Selecionando o padrão.
            d = D(j,:); % Selecionando o valor de saída desejado.

            hiddenExit = x*wHidden;          % Ativação da camada oculta, 
            yHidden = (tanh(hiddenExit))';   % usandO uma função tanh(s).
            exit  = yHidden'*wExit';  % Ativação da camada de saída,
            y        = exit;             % usando uma função linear.
            error     = d - y;               % Erro na saída.

            % Ajuste dos pesos da camada de saída:

            delta_wExit  = error'*alpha*yHidden';
            wExit        = wExit + delta_wExit;

            % Ajuste dos pesos da camada oculta:
            for i = 1:hiddenNeurons
                delta_wHidden = ...
                alpha*error*wExit(:,i)*(1-(yHidden(i).^2))*x;
                wHidden(:,i)       = wHidden(:,i) + delta_wHidden';
            end
        end
        
        % Fim da época. Ao fim da época, os pesos são testados para que se
        % conheça o erro quadrático médio da época.

        Y          = wExit*tanh(X*wHidden)'; % Saída da Rede;
        
        E          = D - Y';                    % Erros calculados;
        
        MSE(period) = sum(sum((E' * E)^0.5))/samples;          % Erro Quadrático Médio.

        % Critério de parada [Erro Mínimo Alcançado]
        if MSE(period) < 0.001
            convergence = 1;
            break 
        end

    end
end
%Iniciando os tests
Dtest = test(:,6:8);
Xtest = test(:,1:5);
errotest = 0;
Ytest = zeros(30,3);
for j = 1:30
    x = Xtest(j,:);
    d = Dtest(j,:);
    
    hiddenExit = x*wHidden;
    yHidden = (tanh(hiddenExit))';
    exit  = yHidden'*wExit';
    y        = exit;
    for i = 1:3
        if(y(:,i)> 0)
           Ytest(j,i) = 1;
        else
          Ytest(j,i) = 0;
        end
    end
    if(sum(Ytest(j,:))> 1)
       if y(1) > y(2) && y(1) > y(3)
           Ytest(j,:) = [1 0 0];
       elseif y(2) > y(3)
           Ytest(j,:) = [0 1 0];
       else
           Ytest(j,:) = [0 0 1];
       end
    end
    error(j,:) = Dtest(j,:)-Ytest(j,:);
    e = sum(abs(error(j,:)));
    if(e ~= 0)
        errotest = errotest + 1;
    end 
end
errotest