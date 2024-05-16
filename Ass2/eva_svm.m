function performance = eva_svm(prediction, labels)

img0 = labels(prediction == 0);
img1 = labels(prediction == 1);


performance = zeros(1,4); 
performance(1,1) = sum(img0 == 0);
performance(1,2) = sum(img0 == 1);
performance(1,3) = sum(img1 == 1);
performance(1,4) = sum(img1 == 0);

end