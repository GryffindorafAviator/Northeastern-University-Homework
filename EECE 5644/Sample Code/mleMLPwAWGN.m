function dummyOut = mleMLPwAWGN(N)

% Maximum likelihood training of a 2-layer MLP
% assuming additive (white) Gaussian noise
close all, 
dummyOut = 0;
% Input N specifies number of training samples

% Generate data using a Gaussian Mixture Distribution
%mu = [1 2;-7 0];
%Sigma = cat(3,[4 0.9; 0.9 0.5],[5 0; 0 0.25]);
%mixp = ones(1,2)/2;
%gm = gmdistribution(mu,Sigma,mixp);
%data = [random(gm,N)';sqrt(0.2)*randn(1,N)];
%figure(1), clf, plot3(data(1,:),data(2,:),data(3,:),'.'); axis equal,
%X = [data(1,:);data(3,:)]; Y = data(2,:);

% Determine/specify sizes of parameter matrices/vectors
nX = 2;%size(X,1); 
nPerceptrons = 5; 
nY = 1;%size(Y,1);
sizeParams = [nX;nPerceptrons;nY]

X = 10*randn(nX,N);
paramsTrue.A = 0.3*rand(nPerceptrons,nX)
paramsTrue.b = 0.3*rand(nPerceptrons,1);
paramsTrue.C = 0.3*rand(nY,nPerceptrons);
paramsTrue.d = 0.3*rand(nY,1);
Y = mlpModel(X,paramsTrue)+1e-5*randn(nY,N);
vecParamsTrue = [paramsTrue.A(:);paramsTrue.b;paramsTrue.C(:);paramsTrue.d];
figure(1), clf, plot3(X(1,:),X(2,:),Y(1,:),'.g');

% Initialize model parameters
params.A = zeros(nPerceptrons,nX);
params.b = zeros(nPerceptrons,1);
params.C = zeros(nY,nPerceptrons);
params.d = mean(Y,2);%zeros(nY,1); % initialize to mean of y
%params = paramsTrue;
vecParamsInit = [params.A(:);params.b;params.C(:);params.d];
%vecParamsInit = vecParamsTrue; % Override init weights with true weights

% Optimize model
vecParams = fminsearch(@(vecParams)(objectiveFunction(X,Y,sizeParams,vecParams)),vecParamsInit);

% Visualize model output for training data
params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
H = mlpModel(X,params);
figure(2), clf, plot(Y,H,'.'); axis equal,
xlabel('Desired Output'); ylabel('Model Output');
title('Model Output Visualization For Training Data')
vecParamsFinal = [params.A(:);params.b;params.C(:);params.d];
figure(1), hold on, plot3(X(1,:),X(2,:),H(1,:),'.r');
xlabel('X_1'), ylabel('X_2'), zlabel('Y and H'),
[vecParamsTrue,vecParamsInit,vecParamsFinal]
keyboard,

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

%
function H = mlpModel(X,params)
N = size(X,2);                          % number of samples
nY = length(params.d);                  % number of outputs
U = params.A*X + repmat(params.b,1,N);  % u = Ax + b, x \in R^nX, b,u \in R^nPerceptrons, A \in R^{nP-by-nX}
Z = activationFunction(U);              % z \in R^nP, using nP instead of nPerceptons
V = params.C*Z + repmat(params.d,1,N);  % v = Cz + d, d,v \in R^nY, C \in R^{nY-by-nP}
H = V; % linear output layer activations
%H = exp(V)./repmat(sum(exp(V),1),nY,1); % softmax nonlinearity for second/last layer
% Add softmax layer to make this a model for class posteriors
%
function out = activationFunction(in)
%out = 1./(1+exp(-in)); % logistic function
out = in./sqrt(1+in.^2); % ISRU



