% --------------------------- TS_PV ---------------------------------------
% Determina la cantidad optima de regresores y clusters ára el modelo TS.
% Asume la existencia de 3 conjuntos de salida y, para constuir vectores de
% regresores, y con ellos la matriz de datos de entrada.
% -------------------------------------------------------------------------

%% Definicion de regresores --

Ny = 30; % 30 por defecto


%% Train
N_train = length(y_train);
% matriz vacia, para guardar datos de entrada train
X_train = zeros(N_train - Ny, Ny);
% cada fila de X_train es de la forma:
% [y(k) y(k-1) y(k-2) ... y(k-Ny)]

% y_train cortado para no repetir datos con regresores
Y_train = y_train(Ny + 1:end);
% plot(Y_train') % grafifo de prueba
% hold on

for i=1:Ny
    X_train(:,i) = y_train(Ny - (i - 1):N_train-i)'; % regresor i-esimo
%     plot(X_train(:,i))
end

% legend('y(k)','y(k-1)','y(k-2)','y(k-3)')

%% Test
N_test = length(y_test);
% matriz vacia, para guardar datos de entrada train
X_test = zeros(N_test - Ny, Ny);
% cada fila de X_test es de la forma:
% [y(k) y(k-1) y(k-2) ... y(k-Ny)]

for i=1:Ny
    X_test(:,i) = y_test(Ny - (i - 1):N_test-i)'; % regresor i-esimo
%     plot(X_test(:,i))
%     hold on
end

% y_test cortado para no repetir datos con regresores
Y_test = y_test(Ny + 1:end);

%% Val
N_val = length(y_val);
% matriz vacia, para guardar datos de entrada val
X_val = zeros(N_val - Ny, Ny);
% cada fila de X_val es de la forma:
% [y(k) y(k-1) y(k-2) ... y(k-Ny)]

for i=1:Ny
    X_val(:,i) = y_val(Ny - (i - 1):N_val-i)'; % regresor i-esimo
%     plot(X_test(:,i))
%     hold on
end

% y_val cortado para no repetir datos con regresores
Y_val = y_val(Ny + 1:end);


%% Calculo de numero optimo de clusters y regresores

max_clusters = 11; % 11 por defecto

% Preparar arreglo para identificar regresores
%label_regresores = repelem([""], [Ny]); % arreglo de strings vacios
label_regresores = strings(1,Ny);

for i=1:Ny
    label_regresores(i) = "y(k-" + i + ")";
end

% label_regresores tiene la forma:
% ["y(k-1)", "y(k-2)", "y(k-2)", ..., "y(k-Ny)"]
% al eliminar un regresor del modelo, se eliminará el string
% correspondiente

label_regresores

% matriz vacia para guardar el valor del error de cada iteracion y la
% cantidad de clusters
error_param = [zeros(2, Ny); fliplr(1:Ny)];
% forma: [error_test; clusters; numero_regresores]

% esta matriz guardara el historial de modificaciones de los regresores, la
% columna j-esima corresponde a la j-esima iteracion. Al los regresores
% eliminados se representan con strings vacios
historial_regresores = strings(Ny,Ny);

% la primera columna son todos los regresores
historial_regresores(:,1) = label_regresores

% Copias de datos de entrada auxiliares, para eliminar regresores sin
% perder datos originales
X_test_aux = X_test;
X_train_aux = X_train;

for i=1:Ny % calcular numero optimo de clusters y regresores
    
    % Numero optimo de clusters
    [errtest,errent] = clusters_optimo(Y_test, Y_train, X_test_aux, ...
                                        X_train_aux, max_clusters);

    [M, indx] = min(errtest);
    %N_clusters = indx + 1
    reglas = indx + 1;
    error_param(2,i) = reglas;
    
    % error de test
    err = errortest(Y_train, X_train_aux, Y_test, X_test_aux, reglas);
    error_param(1,i) = err;
    
    % calcular sensibilidad
    [p, sens] = sensibilidad(Y_train, X_train_aux, reglas);
    
    disp("Indice a eliminar: " + p)
    
    
    % Eliminar regresor menos relevante para la siguiente iteracion
    if i ~= Ny
        % Actualizar string con regresores para la proxima iteracion
        historial_regresores(:,i+1) = strrep(historial_regresores(:,i), ...
                                             label_regresores(p),"");
    end
    
    X_train_aux(:,p) = [];
    X_test_aux(:,p) = [];
    label_regresores(p) = [] % eliminar string del regresor en tiempo real

end

%% Grafico de regresores en funcion del numero de regresores

figure()
plot(error_param(3,:), error_param(1,:), 'Linewidth', 2)
ylabel('MSE de test','FontSize',14)
xlabel('Número de regresores','FontSize',14)
title("Error de test en identificación del modelo")
xlim([1 Ny])

% 5 regresores y 7 clusters, sobreviven
% ["y(k-1)", "y(k-14)", "y(k-15)", "y(k-24)", "y(k-25)"]