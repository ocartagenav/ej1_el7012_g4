
regresores_max = 30;     % Editable

regresores = regresores_max;
neuronas_max = 15;
neuronas_min = 5;

x_train = X_train;
x_test = X_test;
e = 1:1:regresores;
v_regresores = 1:1:regresores;
v_neuronas = 1:1:regresores;
m_regresores = zeros(regresores);

while regresores > 0
    error = 500;
    % Neuronas en capa oculta
    for neuronas=neuronas_min:neuronas_max
        clear net
        net = fitnet(neuronas, 'trainlm');  % Define la red
        net.trainParam.epochs = 25;
        net.trainParam.showWindow=0;        % Oculta el cuadro de entrenamiento
        net = train(net,x_train',Y_train'); % Entrena
        y = net(x_test');                   % Predice sobre test
        err = sqrt(mean((y'-Y_test).^2));   % Calcula el error
        % Guarda el modelo con menor error
        if err < error
            clear nn_optima
            error = err;
            n = neuronas;
            nn_optima = net;
        end
    end
    e(regresores) = error;
    v_neuronas(regresores) = n;
    for reg=1:regresores
        m_regresores(regresores,reg) = v_regresores(reg);
    end
    
    % Analisis de sensibilidad para regresores optimos
    [ exported_ann_structure ] = my_ann_exporter(nn_optima);
    %[p, indice] = annsens(exported_ann_structure, x_train');
    y_max = exported_ann_structure.input_ymax;
    y_min = exported_ann_structure.input_ymin;
    x_max = exported_ann_structure.input_xmax;
    x_min = exported_ann_structure.input_xmin;
    input_preprocessed = (y_max-y_min) * (x_train'-x_min) ./ (x_max-x_min) + y_min;
    % Pass it through the ANN matrix multiplication
    y1 = tanh(exported_ann_structure.IW * input_preprocessed + exported_ann_structure.b1);
    y2 = 1 - y1.^2;
    o = (exported_ann_structure.IW)';
    y3 = o.*exported_ann_structure.LW;
    y4 = y3*y2;
    ind = [];
    for  i=1:size(x_train',1)
        ind(i) = mean(y4(:,i))^2+std(y4(:,i))^2;
    end
    p = find(ind == min(ind));
    q = size(p);
    if q(2) == 1
        x_train(:,p)=[];
        x_test(:,p)=[];
        v_regresores(p)=[];
    else
        x_train(:,p(1))=[];
        x_test(:,p(1))=[];
        v_regresores(p(1))=[];
    end
    
    regresores = regresores - 1;
end

x = 2:1:regresores_max;
plot(x, e(2:end))
ylabel('Error de test','FontSize',14)
xlabel('Número de regresores','FontSize',14)
grid
title('Error del modelo según cantidad de regresores usados','FontSize',14)

