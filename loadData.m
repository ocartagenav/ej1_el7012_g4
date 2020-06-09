
load('DatosPV2015.mat');
load('DatosPV2017.mat');

y_train = data2015; % datos train

L_test = ceil(5*length(data2017)/9); % separacion test y val
y_test = data2017(1:L_test); % datos test
y_val = data2017(L_test+1:end); % datos test

Ny = 30;

% Train
N_train = length(y_train);
% matriz vacia, para guardar datos de entrada train
X_train = zeros(N_train - Ny, Ny);
% cada fila de X_train es de la forma:
% [y(k) y(k-1) y(k-2) ... y(k-Ny)]

% y_train cortado para no repetir datos con regresores
Y_train = y_train(Ny + 1:end);
%plot(Y_train')
%hold on

for i=1:Ny
    X_train(:,i) = y_train(Ny - (i - 1):N_train-i)'; % regresor i-esimo
    %plot(X_train(:,i))
end

% Test
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