close all, clear all, clc;

C = 3;  % Number of classes
N_train = 100; % Number of samples
N_test = 10000;
K = 10; % Number of fold
M = 6;  % Number of max GMM order


[dtrain,ltrain] = generateMultiringDataset(C, N_train);
[dtest,ltest] = generateMultiringDataset(C, N_test);

X_train = dtrain;
Y_train = ltrain;

X_test = dtest;
Y_test = ltest;

% Estimate class prior
count_train_P1 = 0;
count_train_P2 = 0;
count_train_P3 = 0;

for i = 1 : N_train
    if Y_train(i) == 1
        count_train_P1 = count_train_P1 + 1;
    elseif Y_train(i) == 2
        count_train_P2 = count_train_P2 + 1;
    else
        count_train_P3 = count_train_P3 + 1;
    end
end

%Divide data into their class
X_C1 = zeros(2, count_train_P1);
X_C2 = zeros(2, count_train_P2);
X_C3 = zeros(2, count_train_P3);

%Iteration index
count_Xtrain_C1 = 1;
count_Xtrain_C2 = 1;
count_Xtrain_C3 = 1;

for i = 1 : N_train
    if Y_train(i) == 1
        X_C1(:, count_Xtrain_C1) = X_train(:, i);
        count_Xtrain_C1 = count_Xtrain_C1 + 1;
    elseif Y_train(i) == 2
        X_C2(:, count_Xtrain_C2) = X_train(:, i);
        count_Xtrain_C2 = count_Xtrain_C2 + 1;
    else
        X_C3(:, count_Xtrain_C3) = X_train(:, i);
        count_Xtrain_C3 = count_Xtrain_C3 + 1;
    end
end

total_P = count_train_P1 + count_train_P2 + count_train_P3;

P1 = count_train_P1 / total_P;
P2 = count_train_P2 / total_P;
P3 = count_train_P3 / total_P;

%order selection for each GMM model
order_class1 = cross_val(X_C1, count_train_P1);
order_class2 = cross_val(X_C2, count_train_P2);
order_class3 = cross_val(X_C3, count_train_P3);

%Parameters of GMM
[alpha1,mu1,Sigma1] = EMforGMM(order_class1, X_C1, count_train_P1);
[alpha2,mu2,Sigma2] = EMforGMM(order_class2, X_C2, count_train_P2);
[alpha3,mu3,Sigma3] = EMforGMM(order_class3, X_C3, count_train_P3);

%Use GMM to approximate class conditonal pdf
class1_pdf = evalGMM(X_test, alpha1, mu1, Sigma1);
class2_pdf = evalGMM(X_test, alpha2, mu2, Sigma2);
class3_pdf = evalGMM(X_test, alpha3, mu3, Sigma3);


%MAP
p_condition = [class1_pdf * P1; class2_pdf * P2; class3_pdf * P3];
[~, estimate_value] = max(p_condition, [], 1);


%compute Perror of Dtest=10000
T = length(find(estimate_value == Y_test));
disp('order of class 1,2,3 is : ');
disp(order_class1);
disp(order_class2);
disp(order_class3);
Perror = 1 - (T / 10000)
%keyboard,



% Estimate class conditional possibility using GMM and EM algorithm
% 






function best_GMM = cross_val(x, N)
% Order-select using cross-validation
% Performs EM algorithm to estimate parameters and evaluete performance
% on each data set B times, with 1 through M GMM models considered
K = 10;
M = 10;
perf_array = zeros(K, M); % Saving space for performance evaluation

% Make divide index
dummy = ceil(linspace(0, N, K+1));
for k = 1 : K
    indPartitionLimits(k, :) = [dummy(k) + 1,dummy(k + 1)];
end

for k = 1 : K
    indValidate = (indPartitionLimits(k, 1) : indPartitionLimits(k, 2));
    dValidate = x(:, indValidate);
   % dValidate(2, :) = X_C1(2, indValidate);% Using folk k as validation set
    %YValidate = Y(:, indValidate);
    
    if k == 1
        indTrain = (indPartitionLimits(k, 2) + 1 : N);
    elseif k == K
        indTrain = (1 : indPartitionLimits(k, 1) - 1);
    else
        indTrain = [1 : indPartitionLimits(k - 1, 2) indPartitionLimits(k + 1, 1) : N];
    end
    
    dTrain = x(:, indTrain);
    %dTrain(2, :) = X_C1(2, indTrain); % using all other folds as training set
    %YTrain = Y(:, indTrain);
    NTrain = length(indTrain); 
    %NValidate = length(indValidate);
    
    % Select GMM parameters for each order
    for m = 1 : M
        % Non−Built−In: run EM algorith to estimate parameters
        %[alpha, mu, sigma] = EMforGMM(m, dTrain, NTrain, dValidate) ;
        
        % Built−In function : run EM algorithm to estimate parameters
         GMModel = fitgmdist(dTrain', m, 'RegularizationValue', 1e-10);
         alpha = GMModel.ComponentProportion;
         mu = (GMModel.mu)';
         sigma = GMModel.Sigma;
        
        % Calculate log−likelihood performance with new parameters
        perf_array(k, m) = sum(log(evalGMM(dValidate, alpha, mu, sigma)));
        %keyboard;
    end
end

% Calculate average performance for each M and find best
avg_perf = sum(perf_array) / K;
best_GMM = find(avg_perf == max(avg_perf), 1);
%displayProgress(1, x, alpha, mu, sigma);
end

function [alpha_e,mu,Sigma] = EMforGMM(M,x,N)
%M:order, x:dataset, N:number of x, Valset: validateset
% Generates N samples from a specified GMM,
% then uses EM algorithm to estimate the parameters
% of a GMM that has the same number of components
% as the true GMM that generates the samples.

delta = 0.2; % tolerance for EM stopping criterion
regWeight = 1e-2; % regularization parameter for covariance estimates
d = size(x,1);

% Initialize the GMM to randomly selected samples
alpha_e = ones(1,M)/M;
%shuffle the dataset 
shuffledIndices = randperm(N);
mu = x(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates

[~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1); % assign each sample to the nearest mean
for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
    Sigma(:,:,m) = cov(x(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
end
%Run EM algorithm
t = 0; 
Converged = 0; % Not converged at the beginning
% while ~Converged
for i=1:10000    %Calculate GMM distribution according to parameters
    for l = 1:M
        temp(l,:) = repmat(alpha_e(l),1,N).*evalGaussian(x,mu(:,l),Sigma(:,:,l));
    end
    plgivenx = temp./sum(temp,1);
    alphaNew = mean(plgivenx,2);
    w = plgivenx./repmat(sum(plgivenx,2),1,N);
    muNew = x*w';
    for l = 1:M
        v = x-repmat(muNew(:,l),1,N);
        u = repmat(w(l,:),d,1).*v;
        SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
    end
    Dalpha = sum(abs(alphaNew-alpha_e'));
    Dmu = sum(sum(abs(muNew-mu)));
    DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
    Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
    alpha_e = alphaNew; mu = muNew; Sigma = SigmaNew;
    t = t+1; 
    i=i+1;
end
end


function gmm = evalGMM(x, alpha, mu, Sigma)
    % Evaluates GMM on the grid based on parameter values given 
    gmm = zeros(1, size(x, 2));
    for m = 1 : length(alpha)
        gmm = gmm + alpha(m) * evalGaussian(x, mu(:, m), Sigma(:, :, m));
    end
end



function [data,labels] = generateMultiringDataset(C,N)

%C = 3;         %numberOfClasses
%N = 100;         %numberOfSamples
% Generates N samples from C ring-shaped 
% class-conditional pdfs with equal priors

% Randomly determine class labels for each sample
thr = linspace(0,1,C+1); % split [0,1] into C equal length intervals
u = rand(1,N); % generate N samples uniformly random in [0,1]
labels = zeros(1,N);
for l = 1:C
    ind_l = find(thr(l)<u & u<=thr(l+1));
    labels(ind_l) = repmat(l,1,length(ind_l));
end

a = [1:C].^3; b = repmat(2,1,C); % parameters of the Gamma pdf needed later
% Generate data from appropriate rings
% radius is drawn from Gamma(a,b), angle is uniform in [0,2pi]
angle = 2*pi*rand(1,N);
radius = zeros(1,N); % reserve space
for l = 1:C
    ind_l = find(labels==l);
    radius(ind_l) = gamrnd(a(l),b(l),1,length(ind_l));
end

data = [radius.*cos(angle);radius.*sin(angle)];

if 1
    colors = rand(C,3);
    figure, clf,
    for l = 1:C
        ind_l = find(labels==l);
        plot(data(1,ind_l),data(2,ind_l),'.','MarkerFaceColor',colors(l,:)); axis equal, hold on,
    end
end
end
%%%

%EM for selected GMM
%[alpha, mu, sigma] = EMforGMM(m, dTrain, NTrain, dValidate);

%Count

% function g = evalGaussian(x, mu, Sigma)
%     % Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
%     [n,N] = size(x);
%     invSigma = inv(Sigma);
%     C = (2 * pi) * (-n/2) * det(invSigma)^(1/2);
%     E = -0.5 * sum((x-repmat(mu, 1, N)).* (invSigma * (x - repmat(mu, 1, N))) ,1);
%     g = C * exp(E); 
% end

% function gmm = evalGMM(x, alpha, mu, Sigma)
%     % Evaluates GMM on the grid based on parameter values given 
%     gmm = zeros(1, size(x, 2));
%     for m = 1 : length(alpha)
%         gmm = gmm + alpha(m) * evalGaussian(x, mu(:, m), Sigma(:, :, m));
%     end
% end

% function [alpha_est, mu, Sigma] = EMforGMM(M, x, N, val_set)    
% % Uses EM algorithm to estimate the parameters of a GMM that has M
% % number of components based on pre−existing training data of size N
% 
% delta = 0.2; % tolerance for EM stopping criterion
% reg_weight = 1e-2; % regularization parameter for covariance estimates 
% d = size(x, 1); % dimensionality of data
% 
% % Start with equal alpha estimates
% alpha_est = ones(1, M) / M;
% 
% % Set initial mu as random M value pairs from data array
% shuffledIndices = randperm(N); 
% mu = x(:, shuffledIndices(1 : M));
% 
% % Assign each sample to the nearest mean
% [~, assignedCentroidLabels] = min(pdist2(mu', x'), [], 1);
% 
% % Use sample covariances of initial assignments as initial covariance estimates
% for m = 1 : M 
%     Sigma(:, :, m) = cov(x(:, find(assignedCentroidLabels == m))') + reg_weight * eye(d, d);
% end

% Run EM algorith until it converges
% t = 0; 
%displayProgress(t,x,alpha,mu,Sigma);
% Converged = 0; % Not converged at the beginning

% while ~Converged
% for i = 1 : 10000
%     % Calculate GMM distribution according to parameters
%     for l = 1 : M
%         temp(l, :) = repmat(alpha_est(l), 1, N) .* evalGaussian(x, mu(:, l),Sigma(:, :, l));
%     end
%     
%     pl_given_x = temp ./ sum(temp, 1);
%     alpha_new = mean(pl_given_x, 2);
%     w = pl_given_x ./ repmat(sum(pl_given_x, 2), 1, N);
%     mu_new = x * w';
%     
%     for l = 1 : M
%         v = x - repmat(mu_new(:, l), 1, N);
%         u = repmat(w(l, :), d, 1) .* v;
%         Sigma_new(:, :, l) = u * v' + reg_weight * eye(d, d); % adding a small regularization term
%     end
    
    % Change in each parameter
%      Dalpha = sum(abs(alpha_new - alpha_est'));
%      Dmu = sum(sum(abs(mu_new - mu)));
%      DSigma = sum(sum(abs(abs(Sigma_new - Sigma))));
     
     % Check if converged
%      Converged = ((Dalpha + Dmu + DSigma) < delta) | (t == 10);
    
    % Update old parameters
%     alpha_est = alpha_new; 
%     mu = mu_new; 
%     Sigma = Sigma_new;
%     log_lik = sum(log(evalGMM(val_set, alpha_est, mu, Sigma)));
%     Converged = (log_lik < -2.3) | (t == 3);
    
%     t = t + 1; 
%     i = i + 1;
%     %displayProgress(t, x, alpha_est, mu, Sigma);
% end
% end

function displayProgress(t,x,alpha,mu,Sigma)
figure(1),
if size(x,1)==2
    subplot(1,2,1), cla,
    plot(x(1,:),x(2,:),'b.'); 
    xlabel('x_1'), ylabel('x_2'), title('Data and Estimated GMM Contours'),
    axis equal, hold on;
    rangex1 = [min(x(1,:)),max(x(1,:))];
    rangex2 = [min(x(2,:)),max(x(2,:))];
    [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2);
    contour(x1Grid,x2Grid,zGMM); axis equal, 
    subplot(1,2,2), 
end
logLikelihood = sum(log(evalGMM(x,alpha,mu,Sigma)));
plot(t,logLikelihood,'b.'); hold on,
xlabel('Iteration Index'), ylabel('Log-Likelihood of Data'),
drawnow; pause(0.1),
end

function [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2)
x1Grid = linspace(floor(rangex1(1)),ceil(rangex1(2)),101);
x2Grid = linspace(floor(rangex2(1)),ceil(rangex2(2)),91);
[h,v] = meshgrid(x1Grid,x2Grid);
GMM = evalGMM([h(:)';v(:)'],alpha, mu, Sigma);
zGMM = reshape(GMM,91,101);
%figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
end


