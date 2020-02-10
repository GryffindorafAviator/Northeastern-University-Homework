% Expected risk minimization with 2 classes
close all,

n = 2;      % number of feature dimensions
N = 10000;   % number of iid samples

% Class 0 parameters
mu(:,1) = [-0.1;0];
Sigma(:,:,1) = [1 0;0 1];

% Class 1 parameters
mu(:,2) = [0.1;0]; 
Sigma(:,:,2) = [1 0;0 1];

p = [0.8,0.2]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space

% Draw samples from each class pdf
total1 = 0; total2 = 0;
for i = 1 : N
    if label(i) == 0
        % Class 0 samples for each gaussian based on their distribution
        x(:,i) = mvnrnd(mu(:,1),Sigma(:,:,1),1)';
        total1 = total1 + 1;
    end
    
    if label(i) == 1
        % Class 1 samples for each gaussian based on their distribution
        x(:,i) = mvnrnd(mu(:,2),Sigma(:,:,2),1)';
        total2 = total2 + 1;
    end
end

% Plot with class labels
figure(2), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on;
plot(x(1,label==1),x(2,label==1),'+'), axis equal;
legend('Class 0','Class 1');
title('Data and their true labels');
xlabel('x_1'), ylabel('x_2');

% calculate discriminant score based on class pdfs
class1pdf = evalGaussian(x,mu(:,2),Sigma(:,:,2));
class0pdf = evalGaussian(x,mu(:,1),Sigma(:,:,1));
discriminantScore = log(class1pdf)-log(class0pdf);

% compare score to threshold to make decisions
sortDisScr = sort(discriminantScore);
threshold = zeros(1,N);
decision = zeros(N,N);
indCount = zeros(4,N);
distance = zeros(1,N);

for i = 1 : N
    if i == 1
        threshold(i) = sortDisScr(1) - 0.02;
        
    elseif i == N
        threshold(i) = sortDisScr(N) - 0.02;
        
    else
        threshold(i) = (sortDisScr(i-1) + sortDisScr(i+1)) / 2;  
    end
end

for i = 1 : N
        decision(:,i) = (discriminantScore >= log(threshold(i)));
end

for i = 1 : N
    ind00 = find(logical(decision(:,i)==0)' & label==0); %p00 = length(ind00)/Nc(0); % probability of true negative
    ind10 = find(logical(decision(:,i)==1)' & label==0); %p10 = length(ind10)/Nc(0); % probability of false positive
    ind01 = find(logical(decision(:,i)==0)' & label==1); %p01 = length(ind01)/Nc(1); % probability of false negative
    ind11 = find(logical(decision(:,i)==1)' & label==1); %p11 = length(ind11)/Nc(1); % probability of true positive
    
    indCount(1:4,i)=[size(ind10,2)/Nc(1),size(ind11,2)/Nc(2),threshold(i),(size(ind10,2)+size(ind01,2))/N];  
end

indCount = (sortrows(indCount',1))';

[m,p] = min(indCount(4,:));
perror = indCount(4,p);

% Plot ROC Curve
figure(1), clf,
plot(indCount(1,:),indCount(2,:)), axis equal;hold on;
%xlim([0 1]), ylim([0 1])
legend('roc curve');
title('ROC Space');
xlabel('FPR'), ylabel('TPR');
plot(indCount(1,p),indCount(2,p),"ro");
