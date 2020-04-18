clear all, close all,

% Generate n-dimensional N iid vector-samples from a Gaussian pdf
% Here the mean vectors are in mu and covariance matrix is Sigma,
% which is generated from a randomly selected matrix A.
n = 2; N = 1000;
mu = 100*ones(n,1);
A = rand(n,n);
Sigma = A*A';
x = A*randn(n,N)+mu*ones(1,N);

% Sample-based estimates of mean vector and covariance matrix
muhat = mean(x,2);
Sigmahat = cov(x');

% Subtract the estimated mean vector to make the data 0-mean
xzm = x - muhat*ones(1,N);

% Get the eigenvectors (in Q) and eigenvalues (in D) of the
% estimated covariance matrix
[Q,D] = eig(Sigmahat);

% Sort the eigenvalues from large to small, reorder eigenvectors
% accordingly as well.
[d,ind] = sort(diag(D),'descend');
Q = Q(:,ind);
D = diag(d);

% Calculate the principal components (in y)
y = Q'*xzm;

% Calculate whitened components (in z)
z = D^(-1/2)*Q'*xzm;

figure(1), 
subplot(3,1,1),plot(x(1,:),x(2,:),'.b'); axis equal,
xlabel('x_1'), ylabel('x_2'),
subplot(3,1,2),plot(y(1,:),y(2,:),'.r'); axis equal,
xlabel('y_1'), ylabel('y_2'),
subplot(3,1,3),plot(z(1,:),z(2,:),'.k'); axis equal,
xlabel('z_1'), ylabel('z_2'),

