% --------------------------- import_PV_data ------------------------------
% Importa datos de generacion fotovoltaica en kW, tiempo de muestreo de
% 1 hora
% -------------------------------------------------------------------------

load('DatosPV2015.mat');
load('DatosPV2017.mat');

y_train = data2015; % datos train

L_test = ceil(5*length(data2017)/9); % separacion test y val
y_test = data2017(1:L_test); % datos test
y_val = data2017(L_test+1:end); % datos test