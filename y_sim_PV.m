% Funcion que simula un modelo TS de generacion PV, a n_steps numero de
% pasos, de 5 auto regresores, [y(k-1), y(k-14), y(k-15), y(k-24), y(k-25)]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Argumentos:
% a: matriz que contiene el inverso de los std de los clusters
% b: centros de los cluster
% g: parametros de las consecuencias
% Nota: a, b y g deben estar en el formato que usa la funcion ysim
% y0: condicion inicial en y, y0 = [y(k-1) y(k-2)...y(k-25)], vector
% columna de 25 componentes, para simular salidas desde el instante k
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Outputs:
% y: vector con los n_steps resultados de las simulaciones, desde y_hat(k)
% hasta y_hat(k + n_steps -1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y_hat = y_sim_PV(a, b, g, y0, n_steps)

% largo del vector de condiciones iniciales, el numero de elementos que
% tiene y0 es igual al regresor más antiguo ocupado, en este caso 25
N_y0 = length(y0);

% vector que agrupa "condiciones iniciales" y predicciones nuevas
% las predicciones nuevas se van guardando en la extensión de ceros
y_full = [y0; zeros(n_steps,1)]; % vector que agrupara ci mas salidas de simulacion

% n_steps iteraciones, una para cada prediccion
for j = (N_y0 + 1):(N_y0 + n_steps)
    
    % toma los datos apropiados de y_full para usar como regresores
    % en la primera iteracion, solo usa datos que se le entregan a la
    % funcion, los que vienen en y0, luego empieza a mezclar y0 con las
    % predicciones calculadas en el camino, segun corresponda
    xj = [y_full(j-1), y_full(j-14), y_full(j-15), y_full(j-24),...
          y_full(j-25)];
      
    yj = ysim(xj, a, b, g); % simulacion (o prediccion) a 1 paso
    
    % llenar vector y_full, para que la prediccion recien calculada pueda
    % ser usada como dato en la siguiente iteracion
    y_full(j) = yj;
    
end

% Tomamos solo los ultimos n_steps datos de y_full como salida, que son
% todos los calculados por esta funcion
y_hat = y_full(N_y0 + 1:end);

end