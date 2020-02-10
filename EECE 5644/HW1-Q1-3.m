close all,

% Generate n-dimensional data vectors from 2 Gaussian pdfs
n = 2; 
N1 = 6358; mu1 = -1*ones(n,1); A1 = 3*(rand(n,n)-0.5); %S1 = A*A';
N2 = 3642; mu2 = 1*ones(n,1); A2 = 2*(rand(n,n)-0.5);
N = N1 + N2;
x1 = A1*randn(n,N1)+mu1*ones(1,N1);
x2 = A2*randn(n,N2)+mu2*ones(1,N2);

% Estimate mean vectors and covariance matrices from samples
mu1hat = mean(x1,2); S1hat = cov(x1');
mu2hat = mean(x2,2); S2hat = cov(x2');

% Calculate the between/within-class scatter matrices
Sb = (mu1hat-mu2hat)*(mu1hat-mu2hat)';
Sw = S1hat + S2hat;

% Solve for the Fisher LDA projection vector (in w)
[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector

% Linearly project the data from both categories on to w
y1 = w'*x1;
y2 = w'*x2;
label1 = zeros(1,N1); label2 = ones(1,N2);
y = [y1, y2; label1, label2];
ySort = sortrows(y',1)';
threshold = zeros(1,N);
decision = zeros(N,N);
indCount = zeros(4,N);

for i = 1 : N
    if i == 1
        threshold(i) = ySort(1,i) - 0.02;
        
    elseif i == N
        threshold(i) = ySort(1,i) - 0.02;
        
    else
        threshold(i) = (ySort(1,i-1) + ySort(1,i+1)) / 2;  
    end
end

for i = 1 : N
        decision(:,i) = (y(1,:) >= threshold(i));
end

for i = 1 : N
    ind00 = find(logical(decision(:,i)')==0 & y(2,:)==0); p00 = length(ind00)/N1; % probability of true negative
    ind10 = find(logical(decision(:,i)')==1 & y(2,:)==0); p10 = length(ind10)/N1; % probability of false positive
    ind01 = find(logical(decision(:,i)')==0 & y(2,:)==1); p01 = length(ind01)/N2; % probability of false negative
    ind11 = find(logical(decision(:,i)')==1 & y(2,:)==1); p11 = length(ind11)/N2; % probability of true positive
    
    indCount(1:4,i)=[size(ind10,2)/N1,size(ind11,2)/N2,threshold(i),(size(ind10,2)+size(ind01,2))/N];  
end

[m,p] = min(indCount(4,:));
perror = indCount(4,p);

% Plot ROC Curve
figure(2), clf,
plot(indCount(1,:),indCount(2,:)), axis equal;hold on;
%xlim([0 1]), ylim([0 1])
legend('roc curve');
title('ROC Space');
xlabel('FPR'), ylabel('TPR');
plot(indCount(1,p),indCount(2,p),"ro");

% Plot the data before and after linear projection
figure(1),
subplot(2,1,1), 
plot(x1(1,:),x1(2,:),'r*'); hold on;
plot(x2(1,:),x2(2,:),'bo'); axis equal, 
subplot(2,1,2), 
plot(y1(1,:),zeros(1,N1),'r*'); hold on;
plot(y2(1,:),zeros(1,N2),'bo'); axis equal,
