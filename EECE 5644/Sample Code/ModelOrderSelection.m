function out = ModelOrderSelection(in)
out = 1;
%%
% clear all, close all, 
L = 5; randompoles = 1.99*(rand(1,L)-0.5);   % poles must be in unit circle
a = poly(randompoles); % True AR filter den. coef.
N = 1e4; % Number of samples
v = randn(1,N); x = filter(1,a,v);
figure(1), clf,
subplot(2,2,1), stem(a), title('A(z) Coefficients for AR Filter 1/A(z)'),
subplot(2,2,3), plot(v), title('Input Signal (White Noise)'),
subplot(2,2,4), plot(x), title('Output Signal (After AR Filtering)'),

MinM = 2; MaxM = 10;
w = zeros(MaxM,MaxM);   d = x(MaxM+1:N); 
for r = 1:MaxM, U(MaxM-r+1,:) = x(r:N-1-MaxM+r); end,
R = U*U'/(N-MaxM);  p = U*d'/(N-MaxM);
% Wiener filter for AR model estimation
for M = MinM:MaxM;
    M,
    RM = R(1:M,1:M);    pM = p(1:M);    
    w(1:M,M) = inv(RM)*pM;  e = d-w(:,M)'*U;   % error samples
    MEANe(M,1) = mean(e);   STDe(M,1) = std(e); C(M,1) = 1/sqrt(2*pi)/STDe(M);
    Neg2lnP = -2*N*log(C(M,1)) + sum(e.^2);
    AIC(M,1) =  Neg2lnP + 2*M;
    AICc(M,1) =  Neg2lnP + 2*M*(N-MaxM)/((N-MaxM)-M-1);
    rho = 1.5; GIC(M,1) =  Neg2lnP + M*(1+rho);
    BIC(M,1) =  Neg2lnP + M*log(N-MaxM);
end
%[MEANe,STDe,MEANe.^2+STDe.^2]
figure(1),
subplot(2,2,2), plot([MinM:MaxM],AIC(MinM:MaxM),'r',[MinM:MaxM],AICc(MinM:MaxM),'b',[MinM:MaxM],GIC(MinM:MaxM),'g',[MinM:MaxM],BIC(MinM:MaxM),'m'); hold on,
xlabel('Model Order'); ylabel('Model Order Selection Objective Value');
subplot(2,2,2), plot([MinM:MaxM],AIC(MinM:MaxM),'r.',[MinM:MaxM],AICc(MinM:MaxM),'b.',[MinM:MaxM],GIC(MinM:MaxM),'y.',[MinM:MaxM],BIC(MinM:MaxM),'m.');
legend('AIC', 'AICc', 'GIC', 'BIC');
%%