% ------------------------- TS_PV_predict ---------------------------------
% Calcula predicciones a 1, 6 y 12 pasos
% -------------------------------------------------------------------------


%% Volver a definir regresores en base a datos originales

% En un comienzo, se dio un maximo de 30 regresores, pero se usan solo
% hasta el 25, por lo que se reconstruyen para no perder datos de los
% ultimos 5 regresores

% regresor mas antiguo usado, calculado luego por TS_PV, se modifica
% manualmente
Ny = 25;
reglas = 6; % deberia ser 7, pero 6 anda bien


% Train
N_train = length(y_train); % datos totales train
% se prepara matriz para los los Ny regresores train
X_train = zeros(N_train - Ny, Ny);

% y_train cortado para no repetir datos con regresores
Y_train = y_train(Ny + 1:end);

for i=1:Ny
    X_train(:,i) = y_train(Ny - (i - 1):N_train-i)'; % regresor i-esimo
end


% Test
N_test = length(y_test); % datos totales test
% se prepara matriz para los los Ny regresores test
X_test = zeros(N_test - Ny, Ny);

% y_test cortado para no repetir datos con regresores
Y_test = y_test(Ny + 1:end);

for i=1:Ny
    X_test(:,i) = y_test(Ny - (i - 1):N_test-i)'; % regresor i-esimo
end


% Val
N_val = length(y_val); % datos totales val
% se prepara matriz para los los Ny regresores val
X_val = zeros(N_val - Ny, Ny);

% y_val cortado para no repetir datos con regresores
Y_val = y_val(Ny + 1:end);

for i=1:Ny
    X_val(:,i) = y_val(Ny - (i - 1):N_val-i)'; % regresor i-esimo
end

% whos X_train
% whos X_test
% whos X_val

%% Entrena modelo con regresores optimos

X_train_opt = [X_train(:,1), X_train(:,14), X_train(:,15), X_train(:,24),...
               X_train(:,25)]; % mantener regresores optimos train
           
X_test_opt = [X_test(:,1), X_test(:,14), X_test(:,15), X_test(:,24),...
               X_test(:,25)];
           
X_val_opt = [X_val(:,1), X_val(:,14), X_val(:,15), X_val(:,24),...
               X_val(:,25)];

% Numero de muestras en cjtos de salida
N_train = length(Y_train);
N_test = length(Y_test);
N_val = length(Y_val);

[model, result] = TakagiSugeno(Y_train, X_train_opt, reglas, [1 2 2]);


%% PREDICCIONES

%% Prediccion a 1 paso

% Train
y_train_1 = ysim(X_train_opt,model.a,model.b,model.g);

% whos y_train_1
% whos Y_train

% figure()
% plot(1:N_train,Y_train)
% hold on
% plot(1:N_train,y_train_1)
% title('Predicción a 1 paso, conjunto de entrenamiento')
% xlabel('Tiempo (horas)','FontSize',14)
% ylabel('Potencia kW','FontSize',14)
% xlim([0 N_train])
% % ylim([-5 5])
% lgd = legend('Datos','Predicción');
% lgd.FontSize = 12;
% lgd.FontWeight = 'Bold';


% Test
y_test_1 = ysim(X_test_opt,model.a,model.b,model.g);

% whos y_test_1
% whos Y_test

% figure()
% plot(1:N_test,Y_test)
% hold on
% plot(1:N_test,y_test_1)
% title('Predicción a 1 paso, conjunto de prueba')
% xlabel('Tiempo (horas)','FontSize',14)
% ylabel('Potencia kW','FontSize',14)
% xlim([0 N_test])
% % ylim([-5 5])
% lgd = legend('Datos','Predicción');
% lgd.FontSize = 12;
% lgd.FontWeight = 'Bold';


% Val% whos y_val_1
% whos Y_val

y_val_1 = ysim(X_val_opt,model.a,model.b,model.g);


figure()
plot(1:N_val,Y_val)
hold on
plot(1:N_val,y_val_1)
title('Predicción a 1 paso, conjunto de validación')
xlabel('Tiempo (horas)','FontSize',14)
ylabel('Potencia (kW)','FontSize',14)
xlim([0 N_val])
% ylim([-5 5])
lgd = legend('Datos','Predicción','Location','northwest');
lgd.FontSize = 12;
lgd.FontWeight = 'Bold';

%% Prediccion a 6 pasos

y_val_6 = zeros(size(Y_val));
y_val_6(1:5) = NaN(5,1);

for i = 25:(length(y_val)-6) % instantes en los cuales hacer predicciones
    y0 = y_val(i - 25 + 1:i);
    %whos y0
    y6 = y_sim_PV(model.a, model.b, model.g, y0, 6);
    %whos y6
    y_val_6(i - 25 + 6) = y6(6); % tomar solo el elemento 6 de la simulacion
end

figure()
plot(1:N_val,Y_val)
hold on
plot(1:N_val,y_val_6)
title('Predicción a 6 pasos, conjunto de validación')
xlabel('Tiempo (horas)','FontSize',14)
ylabel('Potencia (kW)','FontSize',14)
xlim([0 N_val])
% ylim([-5 5])
lgd = legend('Datos','Predicción');
lgd.FontSize = 12;
lgd.FontWeight = 'Bold';

%% Prediccion a 12 pasos

y_val_12 = zeros(size(Y_val));
y_val_12(1:11) = NaN(11,1);

for i = 25:(length(y_val)-12) % instantes en los cuales hacer predicciones
    y0 = y_val(i - 25 + 1:i);
    %whos y0
    y12 = y_sim_PV(model.a, model.b, model.g, y0, 12);
    %whos y12
    y_val_12(i - 25 + 12) = y12(12); % tomar solo el elemento 12 de la simulacion
end

figure()
plot(1:N_val,Y_val)
hold on
plot(1:N_val,y_val_12)
title('Predicción a 12 pasos, conjunto de validación')
xlabel('Tiempo (horas)','FontSize',14)
ylabel('Potencia (kW)','FontSize',14)
xlim([0 N_val])
% ylim([-5 5])
lgd = legend('Datos','Predicción');
lgd.FontSize = 12;
lgd.FontWeight = 'Bold';

%% Prediccion a 2 pasos

y_val_2 = zeros(size(Y_val));
y_val_2(1:1) = NaN(1,1);

for i = 25:(length(y_val)-2) % instantes en los cuales hacer predicciones
    y0 = y_val(i - 25 + 1:i);
    %whos y0
    y2 = y_sim_PV(model.a, model.b, model.g, y0, 2);
    %whos y2
    y_val_2(i - 25 + 2) = y2(2); % tomar solo el elemento 12 de la simulacion
end

figure()
plot(1:N_val,Y_val)
hold on
plot(1:N_val,y_val_2)
title('Predicción a 2 pasos, conjunto de validación')
xlabel('Tiempo (horas)','FontSize',14)
ylabel('Potencia (kW)','FontSize',14)
xlim([0 N_val])
% ylim([-5 5])
lgd = legend('Datos','Predicción');
lgd.FontSize = 12;
lgd.FontWeight = 'Bold';


%% METRICAS DE DESEMPEÑO

%% RMSE

rms_TS = zeros(1,3);

% Metricas val
rms_TS(1) = rmse(Y_val,y_val_1);
rms_TS(2) = rmse(Y_val(6:end),y_val_6(6:end));
rms_TS(3) = rmse(Y_val(12:end),y_val_12(12:end));

rms_TS

%% MAE

mae_TS = zeros(1,3);

% Metricas val
mae_TS(1) = mae(Y_val,y_val_1);
mae_TS(2) = mae(Y_val(6:end),y_val_6(6:end));
mae_TS(3) = mae(Y_val(12:end),y_val_12(12:end));

mae_TS

%% MAPE

mape_TS = zeros(1,3);

% Metricas val
mape_TS(1) = mape(Y_val,y_val_1);
mape_TS(2) = mape(Y_val(6:end),y_val_6(6:end));
mape_TS(3) = mape(Y_val(12:end),y_val_12(12:end));

mape_TS % da infinito, probablemente por los puntos en que los datos se anulan