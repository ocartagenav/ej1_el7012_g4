% RMSE

rmse_nn = zeros(1, 3);

rmse_nn(1,1) = rmse(y_val(regresores:end-1),y_p1(1:end)');
rmse_nn(1,2) = rmse(y_val(6:end),y_p6(1:end)');
rmse_nn(1,3) = rmse(y_val(12:end),y_p12(1:end)');

rmse_nn


% MAE

mae_nn = zeros(1, 3);

mae_nn(1,1) = mae(y_val(regresores:end-1),y_p1(1:end)');
mae_nn(1,2) = mae(y_val(6:end),y_p6(1:end)');
mae_nn(1,3) = mae(y_val(12:end),y_p12(1:end)');

mae_nn


% MAPE

mape_nn = zeros(1, 3);

mape_nn(1,1) = mape(y_val(regresores:end-1),y_p1(1:end)');
mape_nn(1,2) = mape(y_val(6:end),y_p6(1:end)');
mape_nn(1,3) = mape(y_val(12:end),y_p12(1:end)');

mape_nn