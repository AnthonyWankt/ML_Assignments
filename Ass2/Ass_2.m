%% Exercicse 7
load A2_data.mat

pj_data = linear_PCA(train_data_01, train_labels_01); 

%% Exercise 8 

% K = 2
[k2_y, k2_C] = K_means_clustering(train_data_01, 2);

gscatter(pj_data(1,:),pj_data(2,:), k2_y, 'br', 'xo')
set(gca,'FontSize',12)
title('K-means Clustering(K=2)')
xlabel('PC 1')
ylabel('PC 2')
lg = legend('Cluster 1', 'Cluster 2');
lg.FontSize = 10;

% K = 5
[k5_y, k5_C] = K_means_clustering(train_data_01, 5);

figure;
gscatter(pj_data(1,:),pj_data(2,:), k5_y, 'brgcm', 'xo*.s')
set(gca,'FontSize',12)
title('K-mean Clustering(K=5)')
xlabel('PC 1')
ylabel('PC 2')
lg = legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5');
lg.FontSize = 10;

%% Exercise 9

img2_1 = reshape(k2_C(:, 1), [28 28]);
img2_2 = reshape(k2_C(:, 2), [28 28]);

hold on
subplot(1,2,1);
imshow(img2_1);
title('Cluster 1')
subplot(1,2,2);
imshow(img2_2);
title('Cluster 2')
hold off

img5_1 = reshape(k5_C(:, 1), [28 28]);
img5_2 = reshape(k5_C(:, 2), [28 28]);
img5_3 = reshape(k5_C(:, 3), [28 28]);
img5_4 = reshape(k5_C(:, 4), [28 28]);
img5_5 = reshape(k5_C(:, 5), [28 28]);

figure;
hold on
subplot(1,5,1);
imshow(img5_1);
title('Cluster 1')
subplot(1,5,2);
imshow(img5_2);
title('Cluster 2')
subplot(1,5,3);
imshow(img5_3);
title('Cluster 3')
subplot(1,5,4);
imshow(img5_4);
title('Cluster 4')
subplot(1,5,5);
imshow(img5_5);
title('Cluster 5')
hold off

%% Exercise 10

[label_train, cluster_2_C] = K_means_clustering(train_data_01, 2);
[train_per, test_per] = K_means_classifier(label_train, train_labels_01, test_data_01, test_labels_01, cluster_2_C);


%% Exercise 11
% Test different

K_max = 8;
len = length(test_labels_01);

for i = 2:K_max
    
   [label_train, cluster_i_C] = K_means_clustering(train_data_01, i);
   
   [train_i_per, test_i_per] = K_means_classifier(label_train, train_labels_01, test_data_01, test_labels_01, cluster_i_C);
   
end

%% Exercise 12

train_data_transformed = train_data_01';
test_data_transformed = test_data_01';
svm_use = fitcsvm(train_data_transformed, train_labels_01);

train_prediction = predict(svm_use, train_data_transformed);
test_prediction = predict(svm_use, test_data_transformed);

svm_performance_train = eva_svm(train_prediction, train_labels_01);
svm_performance_test = eva_svm(test_prediction, test_labels_01);

%% Exercise 13

beta = linspace(1,6,11)

for i = 1:length(beta)

    gauss_svm = fitcsvm(train_data_01',train_labels_01,'KernelFunction','gaussian','KernelScale', beta(i));

    train_prediction = predict(gauss_svm, train_data_01');
    test_prediction = predict(gauss_svm, test_data_01');

    svm_performance_train = eva_svm(train_prediction, train_labels_01);
    train_misclassification_rate = (svm_performance_train(2) + svm_performance_train(4)) / length(train_prediction)

    svm_performance_test = eva_svm(test_prediction, test_labels_01);
    test_misclassification_rate = (svm_performance_test(2) + svm_performance_test(4)) / length(test_prediction)
    
    if test_misclassification_rate == 0
        break
    end
end