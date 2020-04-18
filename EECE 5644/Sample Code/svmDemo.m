% Link to Matlab documentation on fitcsvm - particularly to the portion
% that describes binary classification with SVMs...
% https://www.mathworks.com/help/stats/fitcsvm.html#bt8v_23-1

clear all, close all,
NperClass=500; % Number of samples for each class 
SeparationBetweenClassMeans = 4;
xTrain = [2*randn(2,NperClass)-SeparationBetweenClassMeans/2*ones(2,NperClass),randn(2,NperClass)+SeparationBetweenClassMeans/2*ones(2,NperClass)];
lTrain = [-ones(1,NperClass),ones(1,NperClass)];
NTrain = 2*NperClass; % Total number of training samples

C = 1e0; % Larger C prefers avoiding constraint violations more than maximizing margin
sigma = 1e0; % Larger sigma results in a smoother boundary
trainedSVM = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','linear');

dTrain = trainedSVM.predict(xTrain')'; % Labels of training data using the trained SVM

figure(1), 
subplot(2,1,1), 
indINCORRECT = find(lTrain.*dTrain == -1); % Find training samples that are incorrectly classified by the trained SVM
plot(xTrain(1,indINCORRECT),xTrain(2,indINCORRECT),'r.'), hold on,
indCORRECT = find(lTrain.*dTrain == 1); % Find training samples that are correctly classified by the trained SVM
plot(xTrain(1,indCORRECT),xTrain(2,indCORRECT),'g.'), axis equal,
title('Training Data (RED: Incorrectly Classified)'),
subplot(2,1,2), stem(dTrain,'.'), xlabel('Sample Index'), ylabel('Decided Label'), title('Classification of Training Data'),
pTrainingError = length(indINCORRECT)/NTrain, % Empirically estimate of training error probability


trainedSVM = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','gaussian','KernelScale',sigma);

dTrain = trainedSVM.predict(xTrain')'; % Labels of training data using the trained SVM

figure(2), 
subplot(2,1,1), 
indINCORRECT = find(lTrain.*dTrain == -1); % Find training samples that are incorrectly classified by the trained SVM
plot(xTrain(1,indINCORRECT),xTrain(2,indINCORRECT),'r.'), hold on,
indCORRECT = find(lTrain.*dTrain == 1); % Find training samples that are correctly classified by the trained SVM
plot(xTrain(1,indCORRECT),xTrain(2,indCORRECT),'g.'), axis equal,
title('Training Data (RED: Incorrectly Classified)'),
subplot(2,1,2), stem(dTrain,'.'), xlabel('Sample Index'), ylabel('Decided Label'), title('Classification of Training Data'),
pTrainingError = length(indINCORRECT)/NTrain, % Empirically estimate of training error probability


