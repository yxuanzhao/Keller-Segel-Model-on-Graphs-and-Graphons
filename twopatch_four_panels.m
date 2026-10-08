% =========================================================================
% Two-patch Keller-Segel: four-panel summary (dot style, as in the
% public-goods paper figure).  All panels share the same chi sweep.
%
%   (a) chi vs n1, n2          (b) chi vs c1, c2
%   (c) chi vs total n1+n2     (d) chi vs total c1+c2
%
% In the discrete-patch model n_i is a population SIZE and c_i is a
% chemoattractant CONCENTRATION (densities appear only in the graphon
% model), and the axis labels follow that convention.
%
% Dashed lines mark the uniform values (nbar, cbar, 2nbar, 2cbar) and are
% VERTICALLY CENTERED in each panel; dotted vertical line marks chi*.
% =========================================================================
clear; clc; close all;

%% Parameters (paper, Section 6)
etan  = 0.05;  etac = 0.08;
alpha = 0.2;   beta = 0.1;
r     = 0.1;   nbar = 8;
cbar  = alpha*nbar/beta;                                    % = 16
f     = @(n) r*n.*(1 - n/nbar);

chi_star = (beta + etac/2)*(etan + 2*r)/(alpha*etan*nbar);  % = 0.4375

%% Sweep: 30 discrete chi values
chi_vals = linspace(0, 0.8, 30);
Tend     = 2e5;
opts     = odeset('RelTol',1e-8,'AbsTol',1e-10);
y0       = [nbar+0.1, nbar-0.1, cbar, cbar];   % small bias toward patch 1

Y = zeros(numel(chi_vals), 4);                 % [n1 n2 c1 c2] at t = Tend
for i = 1:numel(chi_vals)
    [~, y] = ode45(@(t,y) twoPatchKS(t, y, chi_vals(i), etan, etac, ...
                                     alpha, beta, f), [0 Tend], y0, opts);
    Y(i,:) = y(end,:);
end
n1 = Y(:,1); n2 = Y(:,2); c1 = Y(:,3); c2 = Y(:,4);
Ntot = n1 + n2;  Ctot = c1 + c2;

%% Colors
orange = [232, 74, 39]/255;
navy   = [19, 41, 75]/255;

fig = figure('Color','w','Position',[80,60,1050,780]);
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

% ---------- (a) chi vs n1, n2 ----------
ax = nexttile; hold on; box on; grid on; ax.GridAlpha = 0.15;
yline(nbar, '--', 'Color', [0.55 0.55 0.55], 'LineWidth', 2.6);
xline(chi_star, 'k:', 'LineWidth', 2.6);
plot(chi_vals, n1, 'o', 'MarkerSize', 9, 'MarkerFaceColor', orange, ...
     'MarkerEdgeColor', orange);
plot(chi_vals, n2, 'o', 'MarkerSize', 5, 'MarkerFaceColor', navy, ...
     'MarkerEdgeColor', navy);
centerY(ax, nbar, [n1; n2]);
styleAx(ax, '$\chi$', 'Population size', '$\chi$ vs $n_1,\,n_2$');
legend({'$\bar{n}$', '$\chi^\ast$', '$n_1$', '$n_2$'}, ...
       'Interpreter','latex', 'FontSize', 12, 'Location','southwest');

% ---------- (b) chi vs c1, c2 ----------
ax = nexttile; hold on; box on; grid on; ax.GridAlpha = 0.15;
yline(cbar, '--', 'Color', [0.55 0.55 0.55], 'LineWidth', 2.6);
xline(chi_star, 'k:', 'LineWidth', 2.6);
plot(chi_vals, c1, 'o', 'MarkerSize', 9, 'MarkerFaceColor', orange, ...
     'MarkerEdgeColor', orange);
plot(chi_vals, c2, 'o', 'MarkerSize', 5, 'MarkerFaceColor', navy, ...
     'MarkerEdgeColor', navy);
centerY(ax, cbar, [c1; c2]);
styleAx(ax, '$\chi$', 'Chemoattractant concentration', '$\chi$ vs $c_1,\,c_2$');
legend({'$\bar{c}$', '$\chi^\ast$', '$c_1$', '$c_2$'}, ...
       'Interpreter','latex', 'FontSize', 12, 'Location','southwest');

% ---------- (c) chi vs total n ----------
ax = nexttile; hold on; box on; grid on; ax.GridAlpha = 0.15;
yline(2*nbar, '--', 'Color', orange, 'LineWidth', 2.6);
xline(chi_star, 'k:', 'LineWidth', 2.6);
plot(chi_vals, Ntot, 'o', 'MarkerSize', 7, 'MarkerFaceColor', orange, ...
     'MarkerEdgeColor', orange);
centerY(ax, 2*nbar, Ntot);
styleAx(ax, '$\chi$', 'Population size', '$\chi$ vs total population');
legend({'$2\bar{n}$', '$\chi^\ast$', '$n_1+n_2$'}, ...
       'Interpreter','latex', 'FontSize', 12, 'Location','southwest');

% ---------- (d) chi vs total c ----------
ax = nexttile; hold on; box on; grid on; ax.GridAlpha = 0.15;
yline(2*cbar, '--', 'Color', navy, 'LineWidth', 2.6);
xline(chi_star, 'k:', 'LineWidth', 2.6);
plot(chi_vals, Ctot, 'o', 'MarkerSize', 7, 'MarkerFaceColor', navy, ...
     'MarkerEdgeColor', navy);
centerY(ax, 2*cbar, Ctot);
styleAx(ax, '$\chi$', 'Chemoattractant concentration', ...
        '$\chi$ vs total chemoattractant');
legend({'$2\bar{c}$', '$\chi^\ast$', '$c_1+c_2$'}, ...
       'Interpreter','latex', 'FontSize', 12, 'Location','southwest');

%% Save
exportgraphics(fig, 'twopatch_four_panels.png', 'Resolution', 600);
disp('Saved: twopatch_four_panels.png');

%% Helpers
function centerY(ax, refValue, data)
    % set ylim so that the dashed reference line sits mid-panel
    dev = 1.2*max(abs(data - refValue));
    dev = max(dev, 0.05*refValue);          % minimum visible margin
    ylim(ax, [refValue - dev, refValue + dev]);
end

function styleAx(ax, xlab, ylab, ttl)
    set(ax, 'FontSize', 12, 'LineWidth', 1.2, 'TickLabelInterpreter','latex');
    xlabel(ax, xlab, 'Interpreter','latex', 'FontSize', 15);
    ylabel(ax, ylab, 'Interpreter','latex', 'FontSize', 15);
    title(ax, ttl, 'Interpreter','latex', 'FontSize', 15);
    xlim(ax, [0, 0.8]);
end

%% ODE right-hand side  (paper eqs (2.2)-(2.3), N = 2)
function dydt = twoPatchKS(~, y, chi, etan, etac, alpha, beta, f)
    n1 = y(1); n2 = y(2); c1 = y(3); c2 = y(4);
    P12 = 1/(1 + exp(-chi*(c1 - c2)));   % Fermi: prob. of moving 2 -> 1
    P21 = 1 - P12;
    dn1 = (etan/2)*(P12*n2 - P21*n1) + f(n1);
    dn2 = (etan/2)*(P21*n1 - P12*n2) + f(n2);
    dc1 = (etac/2)*(c2 - c1)/2 + alpha*n1 - beta*c1;
    dc2 = (etac/2)*(c1 - c2)/2 + alpha*n2 - beta*c2;
    dydt = [dn1; dn2; dc1; dc2];
end
