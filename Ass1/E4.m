%% Exercise 4
clear
load('A1_data.mat')

% assign lambda values
omega_hat = lasso_ccd(t, X, 0.1); 
y_hat = Xinterp*omega_hat;
y_data = X*omega_hat;

omega_hat2 = lasso_ccd(t, X, 10); 
y_hat2 = Xinterp*omega_hat2;
y_data2 = X*omega_hat2;

omega_hat3 = lasso_ccd(t, X, 1.5); 
y_hat3 = Xinterp*omega_hat3;
y_data3 = X*omega_hat3;

figure(1)
hold on
scatter(n, t, 30, 'b')
scatter(n, y_data, 30, 'filled')
plot(ninterp, y_hat, 'r')
legend('Real Data Points', 'Synthesized Data Points', 'Reconstruction Line')
legend('Location','best');
xlabel('Time')

figure(2)
hold on
scatter(n, t, 30, 'b')
scatter(n, y_data2, 30, 'filled')
plot(ninterp, y_hat2, 'r')
legend('Real Data Points', 'Synthesized Data Points', 'Reconstruction Line')
legend('Location','best');
xlabel('Time')

figure(3)
hold on
scatter(n, t, 30, 'b')
scatter(n, y_data3, 30, 'filled')
plot(ninterp, y_hat3, 'r')
legend('Real Data Points', 'Synthesized Data Points', 'Reconstruction Line')
legend('Location','best');
xlabel('Time')

non_z_coord = sum(omega_hat~=0); %233
non_z_coord2 = sum(omega_hat2~=0); %9
non_z_coord3 = sum(omega_hat3~=0); %33