clear all, close all, 

% Input N specifies number of training samples

% Determine/specify sizes of parameter matrices/vectors
nX = 1;%size(X,1); 
nPerceptrons = 10; 
nY = 1;%size(Y,1);
sizeParams = [nX;nPerceptrons;nY]

Ntrain = 1000;
Ntest = 10000;
datatrain = exam4q1_generateData(Ntrain);
datatest = exam4q1_generateData(Ntest);

X1train = datatrain(1,:);
X1test = datatest(1,:);

% paramsTrue.A = 0.3*rand(nPerceptrons,nX)
% paramsTrue.b = 0.3*rand(nPerceptrons,1);
% paramsTrue.C = 0.3*rand(nY,nPerceptrons);
% paramsTrue.d = 0.3*rand(nY,1);
X2train = datatrain(2,:);
X2test = datatest(2,:);

       
% Initialize model parameters
%    params.A = zeros(nPerceptrons,nX);
%    params.b = zeros(nPerceptrons,1);
%    params.C = zeros(nY,nPerceptrons);
%    params.d = mean(Y,2);%zeros(nY,1); % initialize to mean of y
params.A = -1 + (1+1)*rand(nPerceptrons,nX);
params.b = -1 + (1+1)*rand(nPerceptrons,1);
params.C = -1 + (1+1)*rand(nY,nPerceptrons);
params.d = -1 + (1+1)*rand(nY,1); % initialize to mean of y
%params.d = mean(Y,2);
%     params.A = randn(nPerceptrons,nX);
%     params.b = randn(nPerceptrons,1);
%     params.C = randn(nY,nPerceptrons);
%     params.d = randn(nY,1); % initialize to mean of y

vecParamsInit = [params.A(:);params.b;params.C(:);params.d];

% Optimize model
options = optimset('MaxFunEval',200000, 'MaxIter', 200000); %optimize
vecParams = fminsearch(@(vecParams)(objectiveFunction(X1train,X2train,sizeParams,vecParams)),vecParamsInit,options);

% Visualize model output for training data
params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);

vecParamsFinal = [params.A(:);params.b;params.C(:);params.d];

[vecParamsInit,vecParamsFinal]

H = mlpModel(X1test,params);
minitestMSE = sum(sum((X2test-H).*(X2test-H),1),2)/Ntest;

figure, clf,
plot(X1test,X2test,'ob');hold on;
plot(X1test,H,'or');hold on;
xlabel('X_1'); ylabel('X_2');
legend('true label','output');

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
objFncValue = sum(sum((Y-H).*(Y-H),1),2)/N;    %MSE
%objFncValue = sum(-sum(Y.*log(H),1),2)/N;
% Change objective function to make this MLE for class posterior modeling

end

function H = mlpModel(X,params)
N = size(X,2);                          % number of samples
nY = length(params.d);                  % number of outputs
U = params.A*X + repmat(params.b,1,N);  % u = Ax + b, x \in R^nX, b,u \in R^nPerceptrons, A \in R^{nP-by-nX}
Z = activationFunction(U);              % z \in R^nP, using nP instead of nPerceptons
V = params.C*Z + repmat(params.d,1,N);  % v = Cz + d, d,v \in R^nY, C \in R^{nY-by-nP}
H = V; % linear output layer activations

%H = exp(V)./repmat(sum(exp(V),1),nY,1);
% Add softmax layer to make this a model for class posteriors
%
end

function out = activationFunction(in)
%out = 1./(1+exp(-in)); % logistic function
%out = in./sqrt(1+in.^2); % ISRU
out = log(1+exp(in)); % softplus
end

function x = exam4q1_generateData(N)

m(:,1) = [-9;-4]; Sigma(:,:,1) = 4*[1,0.8;0.8,1]; % mean and covariance of data pdf conditioned on label 3
m(:,2) = [0;0]; Sigma(:,:,2) = 3*[3,0;0,0.3]; % mean and covariance of data pdf conditioned on label 2
m(:,3) = [8;-3]; Sigma(:,:,3) = 5*[1,-0.9;-0.9,1]; % mean and covariance of data pdf conditioned on label 1
componentPriors = [0.3,0.5,0.2]; thr = [0,cumsum(componentPriors)];
%N = 1000; 
u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
%figure(1),clf, %colorList = 'rbg';
for l = 1:3
    indices = find(thr(l)<=u & u<thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices));
    x(:,indices) = mvnrnd(m(:,l),Sigma(:,:,l),length(indices))';
%    figure(1), plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
end
figure, plot(x(1,:),x(2,:),'.'),
xlabel('X_1'); ylabel('X_2');
end