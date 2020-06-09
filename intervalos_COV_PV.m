% ---------------------- intervalos_COV_PV --------------------------------
% Calcula intervalos de predicción de modelos TS en base a metodo de
% Covarianza. Asume la existencia de ambos modelos en el Workspace.
% -------------------------------------------------------------------------


%% TS

%% Matrices Psi train

[y_train_h, h_train, y_reglas_train] = ysim_psi(X_train_opt, model.a,...
                                                model.b, model.g, reglas);
% whos y_train_h
% whos Y_train
% h_train es una matriz de Nxreglas, con N el numero de datos de train
%
% h_train = [h_1(Z(1)) h_2(Z(1))...h_Nr(Z(1))]
% el elemento ij de h_train representa la funcion de activacion de la regla
% j-esima en el dato i-esimo

N_train = length(Y_train); % numero de datos de train

Z_train = [ones(1,N_train); X_train_opt']; % convertir a formato de matriz Psi_j
% fila extra de unos al principio, por el g0
%
% Z_train = [1; y(k-1); y(k-2); u(k-1); u(k-2)]
% cada regresor es en realidad un vector fila de largo N con los datos
% train de ese regresor. 1 es por el termino constante de cada regla.
% theta_0

Psi_train = zeros([size(Z_train) reglas]);
% Psi_train este arreglo 3D almacenara las  matrices Psi_j, de la forma:
%
% Psi_j = [h_j(Z(1))Z(1) | ..... | h_j(Z(N))Z(N)], donde Z(k) es el k-esimo
% vector (columna) de datos de entrenamiento, de tantos componentes como
% regresores tiene el modelo, y h_j(Z(k)) es la funcion de activacion de la
% regla j-esima en el vector Z(k)
%
% El largo de la tercera dimension es igual al numero de reglas del modelo

for j=1:reglas
    Z_aux = zeros(size(Z_train)); %variable auxiliar para ir modificando
    
    % construir matriz Psi_j
    
    for k=1:N_train
        % la columna k-esima es el grado de activacion de la regla j-esima,
        % h_j, evaluado en el dato k-esimo
%         o=h_train(k,j)
%         oo=Z_train(:,k)
%         ooo=o*oo
        Z_aux(:,k) = h_train(k,j)*Z_train(:,k);
    end
    
    Psi_train(:,:,j) = Z_aux;
end

% whos Psi_train
% whos h_train
% whos y_reglas_train


%% Matrices Psi val

[y_val_h, h_val, y_reglas_val] = ysim_psi(X_val_opt, model.a, model.b, ...
                                          model.g, reglas);
% h_val es una matriz de Nxreglas, con N el numero de datos de val
%
% h_val= [h_1(Z(1)) h_2(Z(1))...h_Nr(Z(1))]
% el elemento ij de h_train representa la funcion de activacion de la regla
% j-esima en el dato i-esimo

N_val = length(Y_val); % numero de datos de val

Z_val = [ones(1,N_val); X_val_opt']; % convertir a formato de matriz Psi_j
%
% Z_val = [y(k-1); y(k-2); u(k-1); u(k-2)]
% cada regresor es en realidad un vector fila de largo N con los datos
% val de ese regresor

Psi_val = zeros([size(Z_val) reglas]);
% Psi_val este arreglo 3D almacenara las  matrices Psi_j, de la forma:
%
% Psi_j = [h_j(Z(1))Z(1) | ..... | h_j(Z(N))Z(N)], donde Z(k) es el k-esimo
% vector (columna) de datos de validacion, de tantos componentes como
% regresores tiene el modelo, y h_j(Z(k)) es la funcion de activacion de la
% regla j-esima en el vector Z(k)
%
% El largo de la tercera dimension es igual al numero de reglas del modelo

for j=1:reglas
    Z_aux = zeros(size(Z_val)); %variable auxiliar para ir modificando
    
    % construir matriz Psi_j
    
    for k=1:N_val
        % la columna k-esima es el grado de activacion de la regla j-esima,
        % h_j, evaluado en el dato k-esimo
        Z_aux(:,k) = h_val(k,j)*Z_val(:,k);
    end
    
    Psi_val(:,:,j) = Z_aux;
end

% whos Psi_val
% whos h_val
% whos y_reglas_val


%% Calculo de sigma_j (datos train)

sigmas = zeros(1,reglas);

for j=1:reglas
    % error local, salida real menos salida de regla j
    y_aux = Y_train - y_reglas_train(:,j).*h_train(:,j);
    %y_aux = y_train - y_reglas_train(:,j);
    %y_aux = y_aux.*h_train(:,j); % ponderar por grados de activacion h
    sigmas(j) = var(y_aux,1);
end


%% Covarianza

matrices_varianzas = zeros(N_val,N_val,reglas);

for j=1:reglas
    matrices_varianzas(:,:,j) = sigmas(j)*(eye(N_val)...
        + (Psi_val(:,:,j)'/(Psi_train(:,:,j)*Psi_train(:,:,j)'))*Psi_val(:,:,j));
    
end

%whos matrices_varianzas

%% Matriz de delta_y

delta_y = zeros(N_val,reglas);

%whos delta_y

for j=1:reglas
    for k=1:N_val
        delta_y(k,j) = matrices_varianzas(k,k,j);
    end
end

%whos delta_y
%whos h_val

Its = zeros(N_val,1);

for k=1:N_val
    Its(k) = h_val(k,:)*delta_y(k,:)';
end

%whos Its

%% Intervalo prediccion a 1 paso

alpha = .0613;

y_val_super = y_val_h + alpha*Its;
y_val_infer = y_val_h - alpha*Its;

kas = (1:N_val)';
%kas = kas/2

o = [kas' fliplr(kas')];
whos o
oo = [y_val_super' fliplr(y_val_infer')];
whos oo

figure()
fill(o,oo,[0.9290 0.6940 0.1250],'FaceAlpha',.6,'EdgeColor','none')
hold on
plot(Y_val,'Linewidth', 1,'Color',[0, 0.4470, 0.7410])
title("Intervalo de predicción a 1 paso, validación, \alpha=" + alpha)
xlabel('Tiempo (horas)','FontSize',14)
ylabel('Potencia (kW)','FontSize',14)
xlim([1 N_val])
lgd = legend('Intervalo de 90% de cobertura','Datos de salida y');
lgd.FontSize = 12;
lgd.FontWeight = 'Bold';

enclosed = 0;

for k=1:N_val
    if Y_val(k) >= y_val_infer(k) && Y_val(k) <= y_val_super(k)
        enclosed = enclosed +1;
    end
end

PICP = (enclosed/N_val)*100
PINAW = pinaw(y_val_super,y_val_infer)

%% Intervalo prediccion a 6 pasos

alpha_6 = .194;
Its_6 = Its(6:end);

y_val_super_6 = y_val_6(6:end) + alpha_6*Its_6;
y_val_infer_6 = y_val_6(6:end) - alpha_6*Its_6;

kas_6 = (6:N_val)';
%kas_8 = kas_8(8:end-1);

o_6 = [kas_6' fliplr(kas_6')];
oo_6 = [y_val_super_6' fliplr(y_val_infer_6')];

figure()
fill(o_6,oo_6,[0.9290 0.6940 0.1250],'FaceAlpha',.6,'EdgeColor','none')
hold on
plot(kas_6,Y_val(6:end),'Linewidth', 1,'Color',[0, 0.4470, 0.7410])
title("Intervalo de predicción a 6 pasos, validación, \alpha=" + alpha_6)
xlabel('Tiempo (horas)','FontSize',14)
ylabel('Potencia (kW)','FontSize',14)
xlim([6 N_val])
%ylim([-11 11])
lgd = legend('Intervalo de 90% de cobertura','Datos de salida y');
lgd.FontSize = 12;
lgd.FontWeight = 'Bold';

enclosed_6 = 0;

y_aux_6 = Y_val(6:end);

for k=1:length(y_aux_6)
    if y_aux_6(k) >= y_val_infer_6(k) && y_aux_6(k) <= y_val_super_6(k)
        enclosed_6 = enclosed_6 +1;
    end
end

PICP_6 = (enclosed_6/length(y_aux_6))*100
PINAW_6 = pinaw(y_val_super_6,y_val_infer_6)

%% Intervalo prediccion a 12 pasos

% 12 pasos
alpha_12 = .388;
Its_12 = Its(12:end);

y_val_super_12 = y_val_12(12:end) + alpha_12*Its_12;
y_val_infer_12 = y_val_12(12:end) - alpha_12*Its_12;

kas_12 = (12:N_val)';
%kas_8 = kas_8(8:end-1);

o_12 = [kas_12' fliplr(kas_12')];
oo_12 = [y_val_super_12' fliplr(y_val_infer_12')];

figure()
fill(o_12,oo_12,[0.9290 0.6940 0.1250],'FaceAlpha',.6,'EdgeColor','none')
hold on
plot(kas_12,Y_val(12:end),'Linewidth', 1,'Color',[0, 0.4470, 0.7410])
title("Intervalo de predicción a 12 pasos, validación, \alpha=" + alpha_12)
xlabel('Tiempo (horas)','FontSize',14)
ylabel('Potencia (kW)','FontSize',14)
xlim([12 N_val])
%ylim([-11 11])
lgd = legend('Intervalo de 90% de cobertura','Datos de salida y');
lgd.FontSize = 12;
lgd.FontWeight = 'Bold';

enclosed_12 = 0;

y_aux_12 = Y_val(12:end);

for k=1:length(y_aux_12)
    if y_aux_12(k) >= y_val_infer_12(k) && y_aux_12(k) <= y_val_super_12(k)
        enclosed_12 = enclosed_12 +1;
    end
end

PICP_12 = (enclosed_12/length(y_aux_12))*100
PINAW_12 = pinaw(y_val_super_12,y_val_infer_12)