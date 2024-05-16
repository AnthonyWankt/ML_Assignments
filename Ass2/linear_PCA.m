function projected = linear_PCA(data, labels)

% norm. to zero mean 
data = data - mean(data, 2); 

% SVD
[W_mat,~,~] = svd(data);
W_d = W_mat(:,1:2);

projected = W_d'*data;

% Plot
gscatter(projected(1,:), projected(2,:), labels, 'rb', 'xo')
set(gca,'FontSize',12)
title('2-Dimensional PCA')
xlabel('PC 1')
ylabel('PC 2')
lg = legend('0', '1');
lg.FontSize = 10;
end