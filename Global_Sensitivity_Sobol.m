%% ==========================================================================
% In this script, we generate the global sensitivity of equilibrium
% normalised permeability figure. We use Jansen-type first-order and total-
% order Sobol estimators, with paired-bootstrap 95% percentile intervals.
% Created by Avulundiah Edwin Phiri, Department of Mathematics,
% Copperbelt University, Kitwe, Zambia
% Date: 23/10/26
%% ==========================================================================
clear; clc; close all;rng(20261030, 'twister');
names  = {'r','kappa','lambda','n'};
labels = {'\delta/\gamma','\kappa','\lambda','n'};
deltRange = [0.10, 0.50];   
gamRange  = [0.50, 1.50];   
kapRange  = [0.15, 0.45];
lambRange = [0.20, 0.45];
nRange    = [4.00, 12.0];

rSupport = [deltRange(1)/gamRange(2), ...
            deltRange(2)/gamRange(1)];
bounds = [rSupport; kapRange; lambRange; nRange];
x0 = 0; N = 32768; nBoot = 2000; d = numel(names);
assert(size(bounds,1) == d, 'Names and parameter bounds do not match.');
R0min = 1/(lambRange(2)*(1+kapRange(2)));
assert(R0min > 1, 'Ranges cross R0 = 1. Adjust kappa/lambda bounds.');
fprintf('Guaranteed minimum R0 = %.4f (> 1)\n', R0min);
fprintf('Ratio r support: [%.6f, %.6f] (induced, nonuniform distribution)\n', ...
        rSupport(1), rSupport(2));
PA = zeros(N,d);
PB = zeros(N,d);
PA(:,1) = sampleRatio(N,deltRange,gamRange);
PB(:,1) = sampleRatio(N,deltRange,gamRange);
PA(:,2) = uniformDraw(N,kapRange);
PB(:,2) = uniformDraw(N,kapRange);
PA(:,3) = uniformDraw(N,lambRange);
PB(:,3) = uniformDraw(N,lambRange);
PA(:,4) = uniformDraw(N,nRange);
PB(:,4) = uniformDraw(N,nRange);

YA = equilibriumK(PA,x0);
YB = equilibriumK(PB,x0);
YAB = zeros(N,d);
for j = 1:d
    PC = PA;
    PC(:,j) = PB(:,j);
    YAB(:,j) = equilibriumK(PC,x0);
end

assert(all(isfinite([YA;YB;YAB(:)])), 'Non-finite equilibrium output.');
V = var([YA;YB],1);
assert(V > 0, 'Zero output variance: Sobol indices undefined.');

S1 = zeros(1,d);
ST = zeros(1,d);
for j = 1:d
    S1(j) = 1 - mean((YB-YAB(:,j)).^2)/(2*V);
    ST(j) = mean((YA-YAB(:,j)).^2)/(2*V);
end
S1boot = zeros(nBoot,d);
STboot = zeros(nBoot,d);
for b = 1:nBoot
    idx = randi(N,N,1);
    Vb = var([YA(idx);YB(idx)],1);
    assert(Vb > 0, 'Zero bootstrap variance.');
    for j = 1:d
        S1boot(b,j) = 1 - mean((YB(idx)-YAB(idx,j)).^2)/(2*Vb);
        STboot(b,j) = mean((YA(idx)-YAB(idx,j)).^2)/(2*Vb);
    end
end

S1lo = colQuantile(S1boot,0.025);
S1hi = colQuantile(S1boot,0.975);
STlo = colQuantile(STboot,0.025);
SThi = colQuantile(STboot,0.975);

R = table(string(names(:)), bounds(:,1), bounds(:,2), S1(:), ...
          S1lo(:), S1hi(:), ST(:), STlo(:), SThi(:), ...
    'VariableNames', {'Parameter','SupportMin','SupportMax', ...
                      'FirstOrder','FirstCI_L','FirstCI_U', ...
                      'TotalOrder','TotalCI_L','TotalCI_U'});
disp(R);
fprintf('K*: mean=%.5f, std=%.5f, 5th=%.5f, 95th=%.5f\n', ...
    mean(YA),std(YA),colQuantile(YA,0.05),colQuantile(YA,0.95));

writetable(R,'global_sensitivity_mc_crosscheck.csv');

fig = figure('Color','w','Position',[100,100,930,470]);
bar([S1(:),ST(:)],'grouped');
set(gca,'XTick',1:d,'XTickLabel',labels,'TickLabelInterpreter','tex', ...
    'FontSize',12);
ylabel('Sobol sensitivity index');
legend('First order','Total order','Location','northeast','Box','off');
title('Global sensitivity of equilibrium normalised permeability');
grid on; box on;
try
    exportgraphics(fig,'global_sensitivity_mc_crosscheck.png','Resolution',600);
end
fprintf('Saved global_sensitivity_mc_crosscheck.csv and global_sensitivity_mc_crosscheck.png\n');
function v = uniformDraw(N,range)
    v = range(1) + (range(2)-range(1))*rand(N,1);
end

function r = sampleRatio(N,deltaRange,gammaRange)
    delta = uniformDraw(N,deltaRange);
    gamma = uniformDraw(N,gammaRange);
    r = delta./gamma;
end

function Kstar = equilibriumK(P,x0)
 
    r      = P(:,1);
    kappa  = P(:,2);
    lambda = P(:,3);
    n      = P(:,4);
    c = 1+x0;

    R0 = 1./(lambda.*(1+kappa));
    assert(all(R0>1),'Positive equilibrium requires R0 > 1.');
    sstar = lambda.*kappa./(1-lambda);
    H = r.*(1-sstar)./lambda;
    assert(all(isfinite(H) & H>0),'Expected a positive finite H.');
    lo = zeros(size(H));
    hi = c*ones(size(H));
    for it = 1:65
        mid = 0.5*(lo+hi);
        f = mid - H.*(max(c-mid,0).^n);
        mask = f>0;
        hi(mask) = mid(mask);
        lo(~mask) = mid(~mask);
    end
    xstar = 0.5*(lo+hi);
    Kstar = max(c-xstar,0).^n;
end

function q = colQuantile(X,p)

    assert(p>=0 && p<=1, 'Quantile p must be in [0,1].');
    X = sort(X,1);
    m = size(X,1);
    h = 1+(m-1)*p;
    a = floor(h);
    b = ceil(h);
    t = h-a;
    q = (1-t)*X(a,:) + t*X(b,:);
end
