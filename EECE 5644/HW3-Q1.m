close all, clear all, clc;

C = 3; %numberOfClasses;
%N = 1000; %numberOfSamples;
K = 10; %numberOfFold;

Ntrain = 10000;
[datatrain, labeltrain] = generateMultiringDataset(C, Ntrain);
%[data,labels] = generateMultiringDataset(C,N);


% Maximum likelihood training of a 2-layer MLP
% Input N specifies number of training samples

% Determine/specify sizes of parameter matrices/vectors
nX = 2;%size(X,1); 
nPerceptrons = 7; 
nY = 3;%size(Y,1);
sizeParams = [nX;nPerceptrons;nY];

Xtrain = datatrain;
% Xtest = datatest;

Ytrain = zeros(C,Ntrain);
for i = 1 : Ntrain
    if labeltrain(i) == 1
    Ytrain(1,i) = 1;
    end
    if labeltrain(i) == 2
        Ytrain(2,i) = 1;
    end
    if labeltrain(i) == 3
        Ytrain(3,i) = 1;
    end    
end

% Ytest = zeros(C,Ntrain);
% for i = 1 : Ntrain
%     if labeltest(i) == 1
%     Ytest(1,i) = 1;
%     end
%     if labeltest(i) == 2
%         Ytest(2,i) = 1;
%     end
%     if labeltest(i) == 3
%         Ytest(3,i) = 1;
%     end    
% end

%figure(3), clf, plot3(X(1,:),X(2,:),Y(1,:),'.g');

%keyboard,

NPerror = zeros(1,10);

dummy = ceil(linspace(0,Ntrain,K+1));
for k = 1:K
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
end

 % Initialize model parameters
%     params.A = rand(nPerceptrons,nX);
%     params.b = rand(nPerceptrons,1);
%     params.C = rand(nY,nPerceptrons);
%     params.d = rand(nY,1);
    
  
    %keyboard
    
%     vecParamsInit = [params.A(:);params.b;params.C(:);params.d];

for k = 1 : K
    indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
    dValidate(1,:) = Xtrain(1,indValidate);
    dValidate(2,:) = Xtrain(2,indValidate);% Using folk k as validation set
    YValidate = Ytrain(:,indValidate);
    
    if k == 1
        indTrain = [indPartitionLimits(k, 2) + 1 : Ntrain];
    elseif k == K
        indTrain = [1 : indPartitionLimits(k, 1) - 1];
    else
        indTrain = [1 : indPartitionLimits(k - 1, 2) indPartitionLimits(k + 1, 1) : Ntrain];
    end
    
    dTrain(1, :) = Xtrain(1, indTrain);
    dTrain(2, :) = Xtrain(2, indTrain); % using all other folds as training set
    YTrain = Ytrain(:, indTrain);
    NTrain = length(indTrain); 
    NValidate = length(indValidate);
    
    % Initialize model parameters
     params.A = rand(nPerceptrons,nX);
     params.b = rand(nPerceptrons,1);
     params.C = rand(nY,nPerceptrons);
     params.d = rand(nY,1);
     
  
    %keyboard
    
    vecParamsInit = [params.A(:);params.b;params.C(:);params.d];

    % Optimize model
    options = optimset('MaxFunEval',200000, 'MaxIter', 200000); %optimize
    vecParams = fminsearch(@(vecParams)(objectiveFunction(dTrain,YTrain,sizeParams,vecParams)),vecParamsInit,options);

    % Visualize model output for training data
    params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
    params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
    params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
    params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
    H = mlpModel(dValidate,params);
%     figure(2), clf, plot(Y,H,'.'); axis equal,
%     xlabel('Desired Output'); ylabel('Model Output');
%     title('Model Output Visualization For Training Data')
    vecParamsFinal = [params.A(:);params.b;params.C(:);params.d];
%     figure(3), hold on, plot3(X(1,:),X(2,:),H(1,:),'.r');
%     xlabel('X_1'), ylabel('X_2'), zlabel('Y and H'),
%     [vecParamsInit,vecParamsFinal];
    
    NCorrect = 0;
    NWrong = NValidate;
    
    for i = 1 : NValidate
        if max(H(:,i)) == H(1,i)
            if YValidate(1,i) == 1
                NCorrect = NCorrect + 1;
            end
        elseif max(H(:,i)) == H(2,i)
            if YValidate(2,i) == 1
                NCorrect = NCorrect + 1;
            end
        elseif max(H(:,i)) == H(3,i)
            if YValidate(3,i) == 1
                NCorrect = NCorrect + 1;
            end
        end
    end
    
    NWrong = NWrong - NCorrect;
    
    NPerror(k) = NWrong / NValidate;
                
    %keyboard;   
end

tempPerror = 0;

for i = 1 : 10
    tempPerror = tempPerror + NPerror(i);
end

Perror = tempPerror / K;

function objFncValue = objectiveFunction(X,Y,sizeParams,vecParams)
N = size(X,2); % number of samples
nX = sizeParams(1);
nPerceptrons = sizeParams(2);
nY = sizeParams(3);
params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
H = mlpModel(X,params);
%objFncValue = sum(sum((Y-H).*(Y-H),1),2)/N;
objFncValue = sum(-sum(Y.*log(H),1),2)/N;
% Change objective function to make this MLE for class posterior modeling
end

function H = mlpModel(X,params)
N = size(X,2);                          % number of samples
nY = length(params.d);                  % number of outputs
U = params.A*X + repmat(params.b,1,N);  % u = Ax + b, x \in R^nX, b,u \in R^nPerceptrons, A \in R^{nP-by-nX}
Z = activationFunction(U);              % z \in R^nP, using nP instead of nPerceptons
V = params.C*Z + repmat(params.d,1,N);  % v = Cz + d, d,v \in R^nY, C \in R^{nY-by-nP}
%H = V; % linear output layer activations
H = exp(V)./repmat(sum(exp(V),1),nY,1); % softmax nonlinearity for second/last layer
% Add softmax layer to make this a model for class posteriors
end

function out = activationFunction(in)
%out = 1./(1+exp(-in)); % logistic function
out = in./sqrt(1+in.^2); % ISRU
end
%

