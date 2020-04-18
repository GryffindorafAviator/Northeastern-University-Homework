% ML and MAP parameter estimation using linear and quadratic models for
% data generated using a quadratic function + 0-mean Gaussian noise
%
% Note that 'pause;' has been used frequently while plotting data - you
% might want to consider removing these as per requirement.
%
% 'bluewhitered' is a colormap that I like to use when plotting data that 
% takes positive and negative values, and when zero values are unimportant.
% 
% Aravind H. M. ("Arvin")       email: hmaravind1@gmail.com

clear; close all; clc;
%% Input and initializations
N = 1000;                       % Number of samples
mu = [1;0];                     % Mean - sample
Sigma = [1,0.3;0.3,1];          % Covariance - sample
SigmaV = 2*0.5;                 % V8ariance - 0-mean Gaussian noise
nRealizations = 100;            % Number of realizations for the ensemble analysis
 
gammaArray = 10.^[-10:0.1:5];   % Array of gamma values
% gammaArray = 10.^[ceil(log10(eps)):-ceil(log10(eps))];

A = [0.4 -0.3;-0.3 0.8];        % Coefficients for quadratic terms
b = [-0.14; 0.97];              % Coefficients for linear terms 
c = -2;                         % Constant

% A = rand(2,2)-0.5;
% b = rand(2,1)-0.5;
% c = rand(1,1);

% True parameter array
params = [c;b(1);b(2);A(1,1);A(1,2)+A(2,1);A(2,2)];

%% Generate 2D Gaussian data, compute value of the function
% Draw N samples of x from a Gaussian distribution
x = mvnrnd(mu,Sigma,N)';

% Calculate y: quadratic in x + additive 0-mean Gaussian noise
y = yFunc(x,params) + SigmaV^0.5*randn(1,N);

%% Visualize the surface and data
[x1Grid,x2Grid] = meshgrid(linspace(min(x(1,:)),max(x(1,:)),100),...
    linspace(min(x(2,:)),max(x(2,:)),100));
xGrid = [x1Grid(:),x2Grid(:)]';
yGrid = reshape(yFunc(xGrid,params),size(x1Grid));

figure('units','normalized','outerposition',[0.01 0.04 0.99 0.95]); 
hold on; box on; grid on;
s=surf(x1Grid,x2Grid,yGrid); s.LineStyle = 'none';
xlabel('x1'); ylabel('x2'); zlabel('y');
view(3); colormap(parula); colorbar;
lgnd=legend(['True surface: ',num2str(params','%1.2f ')]);
lgnd.Location = 'northeast';
pause;

d=scatter3(x(1,:),x(2,:),y,10,'m','filled');
lgnd=legend(['True surface: ',num2str(params','%1.2f ')],...
    'Sample data with added Gaussian noise');
pause;

%% Parameter estimation - linear and quadratic models
% Define z vectors for linear and quadratic models
zL = [ones(1,size(x,2)); x(1,:); x(2,:)];
zQ = [ones(1,size(x,2)); x(1,:); x(2,:); x(1,:).^2; x(1,:).*x(2,:); x(2,:).^2];

% Compute z*z^T for linear and quadratic models
for i = 1:N
    zzTL(:,:,i) = zL(:,i)*zL(:,i)';
    zzTQ(:,:,i) = zQ(:,i)*zQ(:,i)';
end

%% ML Parameter estimation
thetaL_ML = sum(zzTL,3)^-1*sum(repmat(y,size(zL,1),1).*zL,2);
thetaQ_ML = sum(zzTQ,3)^-1*sum(repmat(y,size(zQ,1),1).*zQ,2);

%% Plot results - ML: surfaces
yL_ML = reshape(yFunc(xGrid,[thetaL_ML;0;0;0]),size(x1Grid));
yQ_ML = reshape(yFunc(xGrid,thetaQ_ML),size(x1Grid));

figure('units','normalized','outerposition',[0.01 0.04 0.99 0.95]); 
hold on; box on; grid on;
s=scatter3(x(1,:),x(2,:),y,10,'m','filled');
xlabel('x1'); ylabel('x2'); zlabel('y');
view(3); caxis([-1 1]); colormap(jet);
legend(s,'Data');
pause;

sL_ML=surf(x1Grid,x2Grid,yL_ML); sL_ML.LineStyle = 'none';
sL_ML.CData = ones(size(yGrid))*1; sL_ML.FaceAlpha = 0.5;
legend([s,sL_ML],'Data','ML: Linear');
pause;

sQ_ML=surf(x1Grid,x2Grid,yQ_ML); sQ_ML.LineStyle = 'none';
sQ_ML.CData = ones(size(yGrid))*-1; sQ_ML.FaceAlpha = 0.5;
legend([s,sL_ML,sQ_ML],'Data','ML: Linear','ML: Quadratic');
pause;

delete(s); pause;
s=surf(x1Grid,x2Grid,yGrid); s.LineStyle = 'none';
s.CData = ones(size(yGrid))*0; s.FaceAlpha = 0.5;
legend([sQ_ML,sL_ML,s],['L ',num2str([thetaL_ML;0;0;0]','%1.2f ')], ...
    ['Q ',num2str(thetaQ_ML','%1.2f ')],['T ',num2str(params','%1.2f ')]);
pause;

%% MAP Parameter estimation: \theta \sim \mathcal{N}(0,\gamma \mathrm{I})
for i = 1:length(gammaArray)
    gamma = gammaArray(i);
    thetaL_MAP(:,i) = (sum(zzTL,3)+SigmaV/gamma*eye(size(zL,1)))^-1*sum(repmat(y,size(zL,1),1).*zL,2);
    thetaQ_MAP(:,i) = (sum(zzTQ,3)+SigmaV/gamma*eye(size(zQ,1)))^-1*sum(repmat(y,size(zQ,1),1).*zQ,2);
end

%% Plot results - MAP: variation with gamma
clrs = lines(length(params));
figure('units','normalized','outerposition',[0.01 0.04 0.99 0.95]);

ax=subplot(121); hold on; box on; ax=gca; ax.XScale = 'log';
axis([gammaArray(1) gammaArray(end) min([params;thetaL_MAP(:);thetaQ_MAP(:)])-0.5 ...
    max([params;thetaL_MAP(:);thetaQ_MAP(:)])+1]);
p11=plot(gammaArray,repmat(params,1,length(gammaArray)),'--','LineWidth',2); 
xlabel('gamma'); ylabel('parameters'); title('True parameters');
lgnd=legend([p11],'c','b(1)','b(2)','A(1,1)','A(1,2)+A(2,1)','A(2,2)');
lgnd.Location = 'north'; lgnd.Orientation = 'horizontal'; lgnd.NumColumns = 3; box(lgnd,'off');
pause;

set(gca,'ColorOrderIndex',1); p12=plot(gammaArray,thetaL_MAP,'-','LineWidth',2);
title('MAP Parameter estimation: linear model');
lgnd=legend([p11],'c','b(1)','b(2)','A(1,1)','A(1,2)+A(2,1)','A(2,2)');
pause;

ax=subplot(122); hold on; box on; ax=gca; ax.XScale = 'log';
axis([gammaArray(1) gammaArray(end) min([params;thetaL_MAP(:);thetaQ_MAP(:)])-0.5 ...
    max([params;thetaL_MAP(:);thetaQ_MAP(:)])+1]);
p21=plot(gammaArray,repmat(params,1,length(gammaArray)),'--','LineWidth',2);
set(gca,'ColorOrderIndex',1); p22=plot(gammaArray,thetaQ_MAP,'-','LineWidth',2);
xlabel('gamma'); ylabel('parameters'); title('MAP Parameter estimation: quadratic model')
lgnd=legend([p21],'c','b(1)','b(2)','A(1,1)','A(1,2)+A(2,1)','A(2,2)');
lgnd.Location = 'north'; lgnd.Orientation = 'horizontal'; lgnd.NumColumns = 3; box(lgnd,'off');
pause;

%% Plot results - MAP: surfaces
ind = length(gammaArray);
yL_MAP = reshape(yFunc(xGrid,[thetaL_MAP(:,ind);0;0;0]),size(x1Grid));
yQ_MAP = reshape(yFunc(xGrid,thetaQ_MAP(:,ind)),size(x1Grid));

figure('units','normalized','outerposition',[0.01 0.04 0.99 0.95]); 
hold on; box on; grid on;
s=scatter3(x(1,:),x(2,:),y,10,'m','filled');
xlabel('x1'); ylabel('x2'); zlabel('y');
view(3); caxis([-1 1]); colormap(jet);
legend(s,'Data'); title(['$\gamma$ = ',num2str(gammaArray(ind))],'Interpreter','Latex');
pause;

sL_MAP=surf(x1Grid,x2Grid,yL_MAP); sL_MAP.LineStyle = 'none';
sL_MAP.CData = ones(size(yGrid))*1; sL_MAP.FaceAlpha = 0.5;
legend([s,sL_MAP],'Data','MAP: Linear');
pause;

sQ_MAP=surf(x1Grid,x2Grid,yQ_MAP); sQ_MAP.LineStyle = 'none';
sQ_MAP.CData = ones(size(yGrid))*-1; sQ_MAP.FaceAlpha = 0.5;
legend([s,sL_MAP,sQ_MAP],'Data','MAP: Linear','MAP: Quadratic');
pause;

delete(s); pause;
s=surf(x1Grid,x2Grid,yGrid); s.LineStyle = 'none';
s.CData = ones(size(yGrid))*0; s.FaceAlpha = 0.5;
legend([sQ_MAP,sL_MAP,s],['L ',num2str([thetaL_MAP(:,ind);0;0;0]','%1.2f ')], ...
    ['Q ',num2str(thetaQ_MAP(:,ind)','%1.2f ')],['T ',num2str(params','%1.2f ')]);
pause;

%% Plot results: error
fig = figure; fig.Position([1,2]) = [5,200];
fig.Position(3) = 3*fig.Position(3);

ax1=subplot(1,3,1); box on;
pML=pcolor(x1Grid,x2Grid,(yQ_ML-yGrid)/max(abs(yGrid(:))));
pML.EdgeColor = 'none'; colormap(ax1,bluewhitered); colorbar;
xlabel('x1'); ylabel('x2'); title('Quadratic: (yML-yTruth)/max(|yTruth|)');
pause;

ax2=subplot(1,3,2); box on;
pMAP=pcolor(x1Grid,x2Grid,(yQ_MAP-yGrid)/max(abs(yGrid(:))));
pMAP.EdgeColor = 'none'; colormap(ax2,bluewhitered); colorbar;
xlabel('x1'); ylabel('x2'); title('Quadratic: (yMAP-yTruth)/max(|yTruth|)');
pause;

ax3=subplot(1,3,3); box on;
pDiff=pcolor(x1Grid,x2Grid,yQ_MAP-yQ_ML); pDiff.EdgeColor = 'none';
colormap(ax3,bluewhitered); colorbar;
xlabel('x1'); ylabel('x2'); title('Quadratic: yMAP-yML');
pause;

%% MAP parameter estimation for an ensemble set of samples
tic;
clearvars -except params mu Sigma SigmaV gammaArray nRealizations;
[msqError, avMsqError, avPercentError, avAbsPercentError] = deal(zeros(nRealizations,length(gammaArray)));
for n = 1:nRealizations
    N = 10;%randi([20,30]);

    % Draw N samples of x from a Gaussian distribution
    x = mvnrnd(mu,Sigma,N)';

    % Calculate y: quadratic in x + additive 0-mean Gaussian noise
    yTruth{1,n} = yFunc(x,params);
    y = yTruth{1,n} + SigmaV^0.5*randn(1,N);
    zQ = [ones(1,size(x,2)); x(1,:); x(2,:); x(1,:).^2; x(1,:).*x(2,:); x(2,:).^2];

    % Compute z*z^T for linear and quadratic models
    for i = 1:N; zzTQ(:,:,i) = zQ(:,i)*zQ(:,i)'; end
    
    % MAP parameter estimation
    for i = 1:length(gammaArray)
        gamma = gammaArray(i);
        thetaMAP{1,n}(:,i) = (sum(zzTQ,3)+SigmaV/gamma*eye(size(zQ,1)))^-1*sum(repmat(y,size(zQ,1),1).*zQ,2);
        yMAP{1,n}(:,i) = yFunc(x,thetaMAP{1,n}(:,i));
    end
    
    % Mean squared error in y
    msqError(n,:) = N\sum((yMAP{1,n}-repmat(yTruth{1,n}',1,length(gammaArray))).^2,1);
    
    % Average mean squared error of estimated parameters
    avMsqError(n,1:length(gammaArray)) = length(params)\sum((thetaMAP{1,n} - ...
        repmat(params,1,length(gammaArray))).^2);%./repmat(params,1,length(gammaArray))*100,1);
    
    % Mean (over all parameters) of percent-error of estimated parameters
    avPercentError(n,1:length(gammaArray)) = length(params)\sum((thetaMAP{1,n} - ...
        repmat(params,1,length(gammaArray)))./repmat(params,1,length(gammaArray))*100,1);
    
    % Mean (over all parameters) of abs(percent-error) of estimated parameters
    avAbsPercentError(n,1:length(gammaArray)) = length(params)\sum(abs((thetaMAP{1,n} - ...
        repmat(params,1,length(gammaArray)))./repmat(params,1,length(gammaArray))*100),1);
end
toc;

%% Plot results - MAP Ensemble: mean squared error
fig = figure; fig.Position([1,2]) = [50,100];
fig.Position([3 4]) = 1.5*fig.Position([3,4]);
percentileArray = [5,25,50,75,95];

ax = gca; hold on; box on;
prctlMsqError = prctile(avMsqError,percentileArray,1);
p=plot(ax,gammaArray,prctlMsqError,'LineWidth',2);
xlabel('gamma'); ylabel('average mean squared error of parameters'); ax.XScale = 'log';
lgnd = legend(ax,p,[num2str(percentileArray'),...
    repmat(' percentile',length(percentileArray),1)]); lgnd.Location = 'southwest';
pause;

[~,ind] = min(abs(prctlMsqError(3,:)));
plot(ax,gammaArray(ind),prctlMsqError(3,ind),'ro');
lgnd = legend(ax,p,[num2str(percentileArray'),...
    repmat(' percentile',length(percentileArray),1)]); lgnd.Location = 'southwest';
pause;
    
%% Plot results - MAP Ensemble: variation of average percent error in estimated parameters
%{
figure('units','normalized','outerposition',[0.01 0.04 0.99 0.95]);
percentileArray = [5,25,50,75,95];

ax1=subplot(1,2,1); hold on; box on;
prctlError = prctile(avPercentError,percentileArray,1);
p1=plot(ax1,gammaArray,prctlError,'LineWidth',2);
xlabel('gamma'); ylabel('mean(percent error)'); ax1.XScale = 'log';
lgnd = legend(ax1,p1,[num2str(percentileArray'),...
    repmat(' percentile',length(percentileArray),1)]); lgnd.Location = 'northwest';
pause;
[~,ind1] = min(abs(prctlError(3,:)));
plot(ax1,gammaArray(ind1),prctlError(3,ind1),'ko','MarkerSize',10);
lgnd = legend(ax1,p1,[num2str(percentileArray'),...
    repmat(' percentile',length(percentileArray),1)]); lgnd.Location = 'northwest';
pause;

ax2=subplot(1,2,2); hold on; box on;
prctlAbsError = prctile(avAbsPercentError,[5,25,50,75,95],1);
p2=plot(ax2,gammaArray,prctlAbsError,'LineWidth',2);
xlabel('gamma'); ylabel('mean(abs(percent error))'); ax2.XScale = 'log';
pause;
[~,ind2] = min(prctlAbsError(3,:));
plot(ax2,gammaArray(ind2),prctlAbsError(3,ind2),'kx','MarkerSize',10);
pause;

plot(ax2,gammaArray(ind1),prctlAbsError(3,ind1),'ko','MarkerSize',10);
%}

%% Function to calculate y (without noise), given x and parameters
function y = yFunc(x,params)
    A = [params(4),params(5)/2;params(5)/2,params(6)];
    b = [params(2);params(3)];
    c = params(1);
    y = diag(x'*A*x)' + b'*x + c;
end