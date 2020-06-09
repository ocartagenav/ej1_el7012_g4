%--------------- nn_predict ----------------
% Predice con el modelo neuronal utilizando
%  el conjunto de validacion. Es necesario
%  correr "nn.m" previamente. Se asume
%  la existencia de 3 conjuntos de entrada u
%  y tres conjuntos de salida y.
% -----------------------------------------

pasos = 6;                  % Editable
max_regresor = 12;
aux_x_train = X_train;
regresores = [1,2,3,11,12];
neuronas = 12;


% Red optima ------------------------------
x_train = zeros(length(regresores), length(aux_x_train));
for i=1:length(regresores)
    for j=1:length(X_train)
        x_train(i,j) = aux_x_train(j, regresores(i));
    end
end
clear net
net = fitnet(neuronas, 'trainlm');  % Define la red
%net.trainParam.epochs = 25;
net.trainParam.showWindow=0;        % Oculta el cuadro de entrenamiento
net = train(net,x_train,Y_train'); % Entrena


% Variables ---------------------------------

l_val = length(y_val);
x_val = zeros(l_val - max_regresor, max_regresor);
for i=1:max_regresor
    x_val(:,i) = y_val(max_regresor-(i-1):l_val-i)';
end
Y_val = y_val(max_regresor+1:end);
X_val = zeros(length(regresores), length(x_val));
for i=1:length(regresores)
    for j=1:length(x_val)
        X_val(i,j) = x_val(j, regresores(i));
    end
end

regresores = max_regresor;
% Para 1 paso -----------------------------
if pasos == 1
    xi = X_val;
    y_hat = net(xi);

% % Para 8 y 16 pasos -----------------------
else 
    y_hat = zeros(1,l_val-pasos+1);
    for i=1:regresores+pasos
        y_hat(i) = y_val(i);
    end
    for j=regresores:l_val-pasos
        pred = zeros(1, regresores+pasos+1);
        % Casos base ----------------------
        % 1-12
        for i=1:regresores
            pred(i) = y_hat(j-regresores+i);
        end
        
        % Iteracion hasta paso -----------
        for i=regresores+1:regresores+pasos+1
            xi = [pred(i-1), pred(i-2), pred(i-3), pred(i-11), pred(i-12)];
            yi = net(xi');
            pred(i) = yi;
        end
        y_hat(j) = yi;
    end
end

% Graficos -------------------------------
figure()
plot(y_val(regresores:end),'Linewidth', 1)
hold on
plot(y_hat,'Linewidth', 1)
grid
title(sprintf('Predicción a %d pasos, conjunto de validación', pasos))
xlabel('Número de muestra','FontSize',14)
ylabel('Amplitud','FontSize',14)
xlim([0 l_val])
%ylim([-5 5])
lgd = legend('Datos','Predicción');
lgd.FontSize = 12;
lgd.FontWeight = 'Bold';

if pasos == 1
    y_p1 = y_hat;
elseif pasos == 6
    y_p6 = y_hat;
elseif pasos == 12
    y_p12 = y_hat;
end