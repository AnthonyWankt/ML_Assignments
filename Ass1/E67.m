%% Exercise 6

clear
load A1_data.mat

lambda_min = 0.001;

frame_length = size(Xaudio,1);
itr = floor(length(Ttrain) / frame_length); 
max_lmbd = zeros(itr, 1); 

% Cal lambdas
for i = 1:itr
    max_lmbd(i) = max(abs(Xaudio'* Ttrain(1 + frame_length * (i - 1):i * frame_length)));
end
 
lambda_max = max(max_lmbd);
N_lambda = 100;
fold_amount = 3;
grid_lambda = exp(linspace(log(lambda_min), log(lambda_max), N_lambda));

[wopt, lambdaopt, RMSEval, RMSEest] = multiframe_lasso_cv(Ttrain, Xaudio, grid_lambda, fold_amount);

figure(6)
hold on
plot(grid_lambda, RMSEval)
plot(grid_lambda, RMSEest)
xline(lambdaopt,'b')
legend('RMSEval', 'RMSEest', 'Optimal Lambda')
xlabel('Lambda')

figure(7)
hold on
plot(log(grid_lambda), RMSEval)
plot(log(grid_lambda), RMSEest)
xline(log(lambdaopt),'b')
legend('RMSEval', 'RMSEest', 'Optimal Lambda')
xlabel('Lambda')

%% Exercise 7

load A1_data.mat

% lambdaopt = 0.0041;

Ytest = lasso_denoise(Ttest, Xaudio, lambdaopt);
soundsc(Ytest, fs)

save('denoised_audio','Ytest','fs')
