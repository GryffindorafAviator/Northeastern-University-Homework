function [Y,C,labels] = ClusteringWithMeanShiftThenSpectral(N,M)

% Sample call: 
% [Y,C,labels] = ClusteringWithMeanShiftThenSpectral(100,4);
% This demo generates N random 2-dim samples from 
% an order M mixture of Gaussians with arbitrary 
% weights-means-covariances. Uses a fixed-width
% spherically symmetric Gaussian kernel
% based KDE to improve cluster spread using
% mean shift, then uses a basic spectral
% clustering method to assign cluster labels
% selecting number of clusters automatically
% based on eigenvalue sequence in latter.

% Generate iid samples from a random M-GMM
[Y,C] = randGMM(N,M); % Y data, C component labels

% Perform a fixed number of mean shift iterations
T = 6; % Number of mean-shift iterations to be performed
X = meanshift(Y,T);

% Perform basic spectral clustering on X
labels = spectralclustering(X);

nClusters = length(unique(labels));
figure(1), clf, 
for k = 1:nClusters
    plot(Y(1,find(labels==k)),Y(2,find(labels==k)),'.','Color',hsv2rgb([rand,1,1])); 
    title(strcat({'Number of Clusters = '},num2str(nClusters))),
    axis equal, hold on, drawnow, 
end, 

%%%%%%
function labels = spectralclustering(Y)
[D,N] = size(Y);
covY = cov(Y');
sigma2 = (1/D)*trace(covY)*(4/((2*D+1)*N))^(2/(D+4));
pd = pdist2(Y',Y'); S = exp(-0.5*pd.^2/sigma2);
%Deg = diag(sum(S,2));%^(-1/2); 
%Lsym = Deg^(-1/2)*(Deg-S)*Deg^(-1/2); %Graph Laplacian
%[V,D] = eig(eye(N)-Lsym); [d,ind]=sort(diag(D),'descend'); 
[V,D] = eig(S); [d,ind]=sort(diag(D),'descend'); 
D = diag(d); V = V(:,ind);
%figure(2), clf, stem(d,'r.'), hold on, stem(diff(d),'b.'); drawnow,
dGYY = sort(abs(eig(S)),'descend');
nClusters = min(find(abs(cumsum(dGYY)>=0.99*N))); % the sume of evals for GYY will be N
% the smallest number of clusters that total almost N in e.val.
%[~,nClusters]=min(diff(d)); % crude method for deciding on number of clusters
X = V(:,1:nClusters)'; % map data to relevant subspace of eigenfunction injection
% the mapped data X will be clustered by angular separation
[~,labels] = max(abs(X),[],1); % crude final label assignment method
%figure(3), clf, 
%if nClusters == 2
%    plot(X(1,find(labels==1)),X(2,find(labels==1)),'r.'); hold on,
%    plot(X(1,find(labels==2)),X(2,find(labels==2)),'g.');  
%    drawnow,
%elseif nClusters == 3
%    plot3(X(1,find(labels==1)),X(2,find(labels==1)),X(3,find(labels==1)),'r.'); hold on,
%    plot3(X(1,find(labels==2)),X(2,find(labels==2)),X(3,find(labels==2)),'g.'); 
%    plot3(X(1,find(labels==3)),X(2,find(labels==3)),X(3,find(labels==3)),'b.'); 
%    drawnow,
%end,

%%%%%%
function X = meanshift(Y,T)
[D,N] = size(Y); covY = cov(Y');
sigma2 = (1/D)*trace(covY)*(4/((2*D+1)*N))^(2/(D+4));
% sigma2*identity is the covariance used for KDE Gaussians
X = Y; % initialize mean shift iterations to data points
for t = 1:T % you can substitute this with a better stopping criterion
    figure(1), clf, plot(Y(1,:),Y(2,:),'r.');drawnow, 
    axis equal, hold on, plot(X(1,:),X(2,:),'b.'), 
    title(strcat({'Iteration '},num2str(t),{' of '},num2str(T)));
    %pause(3),
    pd = pdist2(X',Y');
    GXY = exp(-0.5*pd.^2/sigma2)/((2*pi)^(D/2)*sqrt(sigma2)^D);
    W = inv(diag(sum(GXY,2)))*GXY;
    X = Y*W';
end
figure(1), clf, plot(Y(1,:),Y(2,:),'r.'); 
axis equal, hold on, plot(X(1,:),X(2,:),'b.'), 
title(strcat({'Iteration '},num2str(t),{' of '},num2str(T))),
%keyboard,

%%%%%%
function [Y,C] = randGMM(N,M)
% Generates N 2-dim samples 
if 0    % random GMM parameters
    temp = rand(1,M); a = temp/sum(temp); % probability for each Gaussian component
    mu = 10*(rand(2,M)-0.5); % means for each Gaussian component
    S = randn(2,2,M); % scale matrices for each Gaussian component
end
if 1    % fixed GMM parameters
    a = ones(1,M)/M;
    mu = [linspace(-M,M,M);zeros(1,M)];
    tempS = 0.25*M/(M-1)*eye(2,2);
    S = repmat(tempS,1,1,M);
end
Z = randn(2,N); % 0-mean I-scale Gaussian samples
Nc = zeros(1,M); temp = rand(1,N); ca = [0,cumsum(a)]; 
for m = 1:M,
    Nc(1,m) = length(find(ca(m)<=temp & temp<ca(m+1)));
end
cNc = [0,cumsum(Nc)]; C = zeros(1,N); Y = zeros(2,N);
for m = 1:M,
    C(cNc(m)+1:cNc(m+1)) = m*ones(1,Nc(m));
    Y(:,cNc(m)+1:cNc(m+1)) = S(:,:,m)*Z(:,cNc(m)+1:cNc(m+1))+mu(:,m)*ones(1,Nc(m));
end