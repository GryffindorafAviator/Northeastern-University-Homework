function mlpSGDdemo

close all,
% Size of MLP
p = 5; n = 2;
% True parameters of MLP that generates the training data
ttrue.b2 = 10*rand; ttrue.W2 = 10*rand(1,p); ttrue.b1 = 10*rand(p,1); ttrue.W1 = 10*rand(p,n);
N = 1000; % number of training input-output pairs
X = randn(n,N); [d,dummy] = mlp(X,ttrue,[1,0]);
data.X = X; data.d = d+1e-6*randn(1,N);
% Randomly initialize parameter estimates
% theta.b2 = rand; theta.W2 = rand(1,p); theta.b1 = rand(p,1); theta.W1 = rand(p,n);
% Initialize model to the vicinity of the true MLP
jitter = 5e-1; theta.b2 = ttrue.b2+jitter*randn; theta.W2 = ttrue.W2+jitter*randn(1,p); theta.b1 = ttrue.b1+jitter*randn(p,1); theta.W1 = ttrue.W1+jitter*rand(p,n);
% Train the MLP with SGD starting from specified initial weights
theta = trainmlp(data,theta);

%%%
% Below, I am assuming that at each iteration we use a single-sample based
% stochastic gradient; will generalize to larger batches later...
function theta = trainmlp(data,theta)
[n,N] = size(data.X);
T = 2*N;
t = 0; i = randi(N,1,T);
while t <= T
    t = t + 1;
    eta = 1e-1/(1+1e-3*t); % reduce step size to asymptotically eliminate residual jitter due to stochastic updates
    xt = data.X(:,i(t)); dt = data.d(1,i(t)); % choose a sample randomly
    [e2t,ge2t] = sqerr(xt,dt,theta,[1,1]);
    figure(1), 
    subplot(1,2,1), plot(t,e2t,'.'), xlim([0,T]), xlabel('Iterations'), ylabel('Instantaneous Squared Error'), hold on, drawnow,
    subplot(1,2,2), semilogy(t,e2t,'.'), xlim([0,T]), xlabel('Iterations'), ylabel('Instantaneous Squared Error'), hold on, drawnow,
    theta.b2 = theta.b2 - eta*ge2t.b2;
    theta.W2 = theta.W2 - eta*ge2t.W2;
    theta.b1 = theta.b1 - eta*ge2t.b1;
    theta.W1 = theta.W1 - eta*ge2t.W1;
end

%%%
% Below, I am assuming that at each iteration we use a single-sample based
% stochastic gradient; will generalize to larger batches later...
function [e2,ge2] = sqerr(x,d,theta,flag)
if flag(1) | flag(2)
	[y,gy] = mlp(x,theta,flag);
    e = (d - y);
    if flag(1),
        e2 = e.^2;
    end
    if flag(2),
        ge2.b2 = -2*e*gy.b2;
        ge2.W2 = -2*e*gy.W2';
        ge2.b1 = -2*e*gy.b1;
        ge2.W1 = -2*e*gy.W1;
    else
        ge2 = NaN;
    end
end

%%%
function [y,gy] = mlp(x,theta,flag)
[n,N] = size(x); m = length(theta.b1);
if flag(1) | flag(2)
	l = theta.W1*x + theta.b1*ones(1,N);   % first linear layer
	[f,gf] = sigmoid(l);  % first nonlinear layer
	y = theta.W2*f + theta.b2*ones(1,N);    % output layer 
	if flag(2)
        gy.b2 = ones(1,N);
        gy.W2 = f;
        gy.b1 = repmat(gy.W2,1,N).*gf;
        gy.W1 = zeros(m,n,N);
        for j = 1:N % bad for loop that needs to be eliminated
            gy.W1(:,:,j) = gy.b1(:,j)*x(:,j)';
        end
    else
        gy = NaN;
	end
end

%%%
function [s,gs] = sigmoid(ksi)
s = 1./(1+exp(-ksi));
gs = s.*(1-s);

    




