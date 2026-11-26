%% ========================================================================
% This script we produce the post-nucleation threshold comparison 
% Created by Avulundiah Edwin Phiri, Department of Mathematics, 
% Copperbelt University
% Date: 21/07/26
%% ========================================================================
clear; close all;
delta=1; gamma=1; kappa=1; n=5; x0=0.10; s0=0.50;
losses=[0.75,0.25]; T=linspace(0,50,1001);
opts=odeset('RelTol',1e-10,'AbsTol',1e-12, ...
    'Events',@(t,y) closureEvent(t,y,x0));
figure('Color','w','Position',[100 100 660 520]);
colors=[0.75 0.18 0.34; 0.08 0.36 0.58];
for j=1:numel(losses)
    lambda=losses(j);
    f=@(t,y) [delta*(1+x0-y(2))^n*(1-y(1)) - gamma*y(1)/(kappa+y(1))*y(2); ...
        (y(1)/(kappa+y(1))-lambda)*y(2)];
    sol=ode45(f,[0 50],[s0 x0],opts);
    if sol.x(end)<50-1e-9
        error('Pore closure occurred before the plot end.');
    end
    vals=deval(sol,T);
    subplot(2,1,1); hold on;
    plot(T,vals(1,:),'LineWidth',2,'Color',colors(j,:));
    subplot(2,1,2); hold on;
    plot(T,vals(2,:),'LineWidth',2,'Color',colors(j,:));
end
subplot(2,1,1);
ylabel('Substrate, s(\tau)'); xlim([0 50]); ylim([0 1.05]); grid on;
legend('R_0=2/3 (depletion)','R_0=2 (persistence)','Location','southeast');
subplot(2,1,2);
xlabel('Dimensionless time, \tau'); ylabel('Mineral loading, x(\tau)');
xlim([0 50]); ylim([0 inf]); grid on;
exportgraphics(gcf,'threshold_dynamics_revised.pdf','ContentType','vector');

function [value,isterminal,direction]=closureEvent(~,y,x0)
value=1+x0-y(2); isterminal=1; direction=-1;
end
