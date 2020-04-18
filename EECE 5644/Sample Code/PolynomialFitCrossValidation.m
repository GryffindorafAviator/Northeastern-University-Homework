function out = PolynomialFitCrossValidation
out = 1;

% LS polynomial fit assessment using cross validation MSE

% Experiment set up
% Do not use large L; van der Monde matrix becomes ill conditioned rapidly
% In the experiments, for large M, training MSE values are high since
% the estimated parameters are bad due to ill conditioned matrix inversion
% In general, for polynomial model fits, this basis choice is bad.
% Instead use other polynomial bases...
% https://arxiv.org/abs/1504.02118
L = 7; sigma = 3e-3; % True model is of order L; sigma controls AWGN std
N = 50; K = round(sqrt(N)); % N samples and K fold cross validation

% Generate data
v = [randn(L,1);1];
x = 10*randn(1,N); n = sigma*randn(1,N);
PsiX = formPsiX(x,L,N); % Form the necessary transposed van der Monde matrix
y = v'*PsiX + n;

% Divide the data set into K approximately-equal-sized partitions
dummy = ceil(linspace(0,N,K+1));
for k = 1:K
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
end

% Allocate space
MSEtrain = zeros(K,N); MSEvalidate = zeros(K,N); 
AverageMSEtrain = zeros(1,N); AverageMSEvalidate = zeros(1,N);

% Try all polynomial orders between 1 (best line fit) and N-1 (big time overfit)
for M = 1:L+3
    %[M,N],
    
    % K-fold cross validation
    for k = 1:K
        indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
        xValidate = x(indValidate); % Using folk k as validation set
        yValidate = y(indValidate);
        if k == 1
            indTrain = [indPartitionLimits(k,2)+1:N];
        elseif k == K
            indTrain = [1:indPartitionLimits(k,1)-1];
        else
            indTrain = [indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1];
        end
        xTrain = x(indTrain); % using all other folds as training set
        yTrain = y(indTrain);
        Ntrain = length(indTrain); Nvalidate = length(indValidate);
        % Train model parameters
        [wML,MSEtrain(k,M)] = fitPolynomial(M,xTrain, yTrain, Ntrain);
        PsiXvalidate = formPsiX(xValidate,M,Nvalidate);
        MSEvalidate(k,M) = calculateMSE(yValidate,wML,PsiXvalidate);
    end
    AverageMSEtrain(1,M) = mean(MSEtrain(:,M)); % average training MSE over folds
    AverageMSEvalidate(1,M) = mean(MSEvalidate(:,M)); % average validation MSE over folds
end

figure(1), clf,
semilogy(AverageMSEtrain,'.b'); hold on; semilogy(AverageMSEvalidate,'rx');
xlabel('Model Polynomial Order'); ylabel(strcat('MSE estimate with ',num2str(K),'-fold cross-validation'));
legend('Training MSE','Validation MSE');


%%%
function MSE = calculateMSE(y,w,PsiX)
    MSE = mean((y-w'*PsiX).^2);

%%%
function [wML,MSE] = fitPolynomial(M,xTrain, yTrain, Ntrain)
% Fit order-M polynomial usins training data set with MLE method which
% reduces to least-squares curve fitting here due to additive Gaussian
% noise in the model.
% Model: y = PsiX'*w
% LS estimate: wLS = inv(PsiX*PsiX')*PsiX*y
PsiXtrain = formPsiX(xTrain,M,Ntrain);
R = PsiXtrain*PsiXtrain'; q = PsiXtrain*yTrain';
wML = inv(R)*q;
MSE = calculateMSE(yTrain,wML,PsiXtrain); % calculate training MSE at optimal solution

%%%
function PsiX = formPsiX(x,M,N);
% Forms a transposed van der Monde matrix with N values in x up to
% polynomial power M
PsiX = zeros(M+1,N); PsiX(1,:) = ones(1,N);
for m = 1:M, PsiX(m+1,:) = x.^m; end,