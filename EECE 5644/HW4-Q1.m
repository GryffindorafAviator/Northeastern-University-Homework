close all, clear all, clc;

N = 1000;
X_train = exam4q1_generateData(N);

X1_train = X_train(1, :);
X2_train = X_train(2, :);

K = 10; %numberOfFold;
n_X1 = 1; %size(X,1); 
n_Perceptrons = 4; 
n_X2 = 1; %size(Y,1);
sizeParams = [n_X1; n_Perceptrons; n_X2];

% Calculate divide index 
dummy = ceil(linspace(0, N, K+1));

for k = 1 : K
    indPartitionLimits(k, :) = [dummy(k) + 1, dummy(k + 1)];
end
    
% Allocate space
%MSEtrain = zeros(K,N); 
MSEvalidate = zeros(1, K); 
%AverageMSEtrain = zeros(1,N); 
AverageMSEvalidate = zeros(1, N);

% K-fold cross validation
for k = 1 : K
    indValidate = (indPartitionLimits(k, 1) : indPartitionLimits(k, 2));
    X1_Validate = X1_train(indValidate); % Using folk k as validation set
    X2_Validate = X2_train(indValidate);

    if k == 1
        indTrain = (indPartitionLimits(k, 2) + 1 : N);
    elseif k == K
        indTrain = (1 : indPartitionLimits(k, 1) - 1);
    else
        indTrain = (indPartitionLimits(k - 1, 2) + 1 : indPartitionLimits(k + 1, 1) - 1);
    end

    X1_Train = X1_train(indTrain); % using all other folds as training set
    X2_Train = X2_train(indTrain);
    N_train = length(indTrain); 
    N_validate = length(indValidate);

    % Initialize model parameters
    params.A = rand(n_Perceptrons, n_X1);
    params.b = rand(n_Perceptrons, 1);
    params.C = rand(n_X2, n_Perceptrons);
    params.d = rand(n_X2, 1);

    vecParamsInit = [params.A(:); params.b; params.C(:); params.d];

    % Optimize model
    options = optimset('MaxFunEval', 200000, 'MaxIter', 200000); %optimize
    vecParams = fminsearch(@(vecParams)(objectiveFunction(X1_Train, X2_Train, sizeParams, vecParams)), vecParamsInit, options);

    % Visualize model output for training data
    params.A = reshape(vecParams(1 : n_X1 * n_Perceptrons), n_Perceptrons, n_X1);
    params.b = vecParams(n_X1 * n_Perceptrons + 1 : (n_X1 + 1) * n_Perceptrons);
    params.C = reshape(vecParams((n_X1 + 1) * n_Perceptrons + 1 : (n_X1 + 1 + n_X2) * n_Perceptrons), n_X2, n_Perceptrons);
    params.d = vecParams((n_X1 + 1 + n_X2) * n_Perceptrons + 1 : (n_X1 + 1 + n_X2) * n_Perceptrons + n_X2);
    H = mlpModel(X1_Validate, params);
    
    MSEvalidate(k) = calculateMSE(H, X2_Validate);
%     figure(2), clf, plot(Y,H,'.'); axis equal,
%     xlabel('Desired Output'); ylabel('Model Output');
%     title('Model Output Visualization For Training Data')
    vecParamsFinal = [params.A(:); params.b; params.C(:); params.d];
%     figure(3), hold on, plot3(X(1,:),X(2,:),H(1,:),'.r');
%     xlabel('X_1'), ylabel('X_2'), zlabel('Y and H'),
%     [vecParamsInit,vecParamsFinal];

    % Train model parameters
%     [wML,MSEtrain(k, M)] = fitPolynomial(M,xTrain, yTrain, Ntrain);
%     PsiXvalidate = formPsiX(xValidate,M,Nvalidate);
%     MSEvalidate(k, M) = calculateMSE(yValidate,wML,PsiXvalidate);
end

%AverageMSEtrain(1, M) = mean(MSEtrain(:, M)); % average training MSE over folds
AverageMSEvalidate = mean(MSEvalidate(:)); % average validation MSE over folds

function MSE = calculateMSE(X2_estimate, X2)
    MSE = mean((X2_estimate - X2) .^ 2);
end

function objFncValue = objectiveFunction(X1, X2, sizeParams, vecParams)
    %N = size(X, 2); % number of samples
    n_X1 = sizeParams(1);
    n_Perceptrons = sizeParams(2);
    n_X2 = sizeParams(3);
    params.A = reshape(vecParams(1 : n_X1 * n_Perceptrons), n_Perceptrons, n_X1);
    params.b = vecParams(n_X1 * n_Perceptrons + 1 : (n_X1 + 1) * n_Perceptrons);
    params.C = reshape(vecParams((n_X1 + 1) * n_Perceptrons + 1 : (n_X1 + 1 + n_X2) * n_Perceptrons), n_X2, n_Perceptrons);
    params.d = vecParams((n_X1 + 1 + n_X2) * n_Perceptrons + 1 : (n_X1 + 1 + n_X2) * n_Perceptrons + n_X2);
    H = mlpModel(X1, params);
    %objFncValue = sum(sum((Y-H).*(Y-H),1),2)/N;
    %objFncValue = sum(-sum(Y.*log(H),1),2)/N;
    objFncValue = calculateMSE(H, X2);
    % Change objective function to make this MLE for class posterior modeling
end

function H = mlpModel(X1, params)
    N = size(X1, 2);                          % number of samples
    %nY = length(params.d);                  % number of outputs
    U = params.A * X1 + repmat(params.b, 1, N);  % u = Ax + b, x \in R^nX, b,u \in R^nPerceptrons, A \in R^{nP-by-nX}
    Z = activationFunction(U);              % z \in R^nP, using nP instead of nPerceptons
    H = params.C * Z + repmat(params.d, 1, N);  % v = Cz + d, d,v \in R^nY, C \in R^{nY-by-nP}
    %H = V; % linear output layer activations
    %H = exp(V) ./ repmat(sum(exp(V), 1), nY, 1); % softmax nonlinearity for second/last layer
    % Add softmax layer to make this a model for class posteriors
end

function out = activationFunction(in)
%out = 1./(1+exp(-in)); % logistic function
%out = in./sqrt(1+in.^2); % ISRU
out = log(1 + exp(in)); % softplus
end
