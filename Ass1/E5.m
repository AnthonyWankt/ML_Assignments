%% Exercise 5   
clear
load('A1_data.mat')

lambda_min = 0.1;
lambda_max = max(abs(X'*t));
N_lambda = 200;
N_folds = 10;

% Lambda grid
grid_lambda = exp(linspace(log(lambda_min), log(lambda_max), N_lambda));

% Cal optimal w, optimal lambda, and Root mean squared errs
[wopt, lambdaopt, RMSEval, RMSEest] = lasso_cv(t, X, grid_lambda, N_folds);
 
optimal_t = Xinterp*wopt;
data_t = X*wopt;

figure(4)
hold on
plot(log(grid_lambda), RMSEval)
plot(log(grid_lambda), RMSEest)
xline(log(lambdaopt), 'b');
legend('RMSEval', 'RMSEest', 'Optimal Lambda')
xlabel('Logarithmic Lambda')


figure(5)
hold on
scatter(n, t, 30, 'b')
scatter(n, data_t, 30, 'filled')
plot(ninterp, optimal_t, 'r')
legend('Real Data Points', 'Synthesized Data Points', 'Reconstruction Line')
xlabel('Amount Data Instances')