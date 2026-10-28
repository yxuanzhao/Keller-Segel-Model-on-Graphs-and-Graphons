%% run_network_ode.m
clear; clc;

% ------------------ basic ------------------
N = 9;

% Adjacency matrix A (example: symmetric random graph)
A = zeros(N);
rho = 0.1;

for i = 1:N
    ip = mod(i, N) + 1;          % i+1 with wrap (N -> 1)
    im = mod(i-2, N) + 1;        % i-1 with wrap (1 -> N)

    A(i, ip) = N/2;
    A(i, im) = N/2;
end

% optional: make sure no self-loops (should already be true)
A(1:N+1:end) = 0;

% ------------------ parameters ------------------
par.N     = N;
par.A     = A;

par.eta_n = 0.05;
par.eta_c = 0.08;

par.rn    = 0.1;     % growth rate
par.n0    = 8.0;     % carrying capacity

par.alpha = 0.2;
par.beta  = 0.1;

w_star = (par.beta + par.eta_c)*(par.eta_n + par.rn)/(par.alpha*par.eta_n*par.n0);
%par.w     = 0.3984580819264639 + 0.1;
par.w     = 0.3984580819264639 + (0.01)^2;

% ------------------ initial conditions ------------------
seed = 43;
rng(seed);

eps_amp = 1e-2;                  % amplitude of small noise
eps_n = eps_amp * randn(N,1);
eps_c = eps_amp * randn(N,1);

n_init = par.n0 + eps_n;
c_star = par.alpha * par.n0 / par.beta;
c_init = c_star + eps_c;

y0 = [n_init; c_init];           % 2N x 1

% ------------------ solve ODE ------------------
T_final = 5000;
tspan = [0, T_final];

opts = odeset('RelTol',1e-7,'AbsTol',1e-9);
%[t, y] = ode45(@(t,y) rhs_network_ode(t,y,par), tspan, y0, opts);
[t, y] = ode45(@(t,y) rhs_limit_cycles(t,y,par), tspan, y0, opts);

n = y(:, 1:N);
c = y(:, N+1:2*N);

% ------------------ plot ------------------
%figure('Position',[100 100 900 650]);
%tl = tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

selected_nodes = [1, 2, 5, 8, 9];
sample_step = 10;
idx = 1:sample_step:length(t);

figure;
plot(t(idx), n(idx,selected_nodes), 'LineWidth', 2);

hold on;
yline(par.n0, '--k', 'LineWidth', 2);
hold off;

grid on;
box on;

xlabel('Time $t$', ...
    'Interpreter','latex', 'FontSize',18);

ylabel('$n_i(t)$', ...
    'Interpreter','latex', 'FontSize',18);

title(sprintf( ...
    'Population dynamics (Number of nodes $N=%d$)', N), ...
    'Interpreter','latex', 'FontSize',20);

legend_labels = arrayfun( ...
    @(i) sprintf('$n_{%d}(t)$',i), ...
    selected_nodes, ...
    'UniformOutput',false);

%legend_labels{end+1} = '$\bar n$';
ylim([0, 16]);

legend(legend_labels, ...
    'Interpreter','latex', ...
    'Location','eastoutside', ...
    'FontSize',14);


% ---------------- t versus c_i(t) ----------------
figure;
plot(t(idx), c(idx,selected_nodes), 'LineWidth', 2);

hold on;
yline(c_star, '--k', 'LineWidth', 2);
hold off;

grid on;
box on;

xlabel('Time $t$', ...
    'Interpreter','latex', 'FontSize',18);

ylabel('$c_i(t)$', ...
    'Interpreter','latex', 'FontSize',18);

title(sprintf( ...
    'Chemical dynamics (Number of nodes $N=%d$)', N), ...
    'Interpreter','latex', 'FontSize',20);

legend_labels = arrayfun( ...
    @(i) sprintf('$c_{%d}(t)$',i), ...
    selected_nodes, ...
    'UniformOutput',false);

ylim([10, 22]);

legend(legend_labels, ...
    'Interpreter','latex', ...
    'Location','eastoutside', ...
    'FontSize',14);

n_final = y(end,1:N).';
c_final = y(end,N+1:2*N).';

x = (0:N-1)'/N;

n_snapshot = n(end, :);     % 1 × N
c_snapshot = c(end, :);     % 1 × N

figure;
plot(1:N, n_snapshot, '-o', 'LineWidth', 2, 'MarkerSize', 8);
yline(par.n0, '--black', 'LineWidth', 2);
ylim([0, 16]);
xlabel('Node index $i$', 'Interpreter','latex', 'FontSize',18);
ylabel('$n_i$', 'Interpreter','latex', 'FontSize',18);
title(sprintf('Snapshot of $n_i$ at $T=%d$ (Nodes number $%d$)', T_final, N), 'Interpreter','latex', 'FontSize',20);
grid on;

figure;
plot(1:N, c_snapshot, '-o', 'LineWidth', 2, 'MarkerSize', 8);
yline(c_star, '--black', 'LineWidth', 2);
ylim([8, 24]);
xlabel('Node index $i$', 'Interpreter','latex', 'FontSize',18);
ylabel('$c_i$', 'Interpreter','latex', 'FontSize',18);
title(sprintf('Snapshot of $c_i$ at $T=%d$ (Nodes number $%d$)', T_final, N), 'Interpreter','latex', 'FontSize',20);
grid on;
