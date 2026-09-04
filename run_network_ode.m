%% run_network_ode_corrected.m
clear; clc;

% ---------------- parameters ----------------
N   = 20;
rho = 0.10;

par.N     = N;
par.eta_n = 0.05;
par.eta_c = 0.08;     % manuscript eta_c; RHS will contain eta_c/2
par.rn    = 0.10;
par.n0    = 8.0;
par.alpha = 0.20;
par.beta  = 0.10;

par.c0    = par.alpha*par.n0/par.beta;

par.w_star = 0.3984580819264639;
eps_bif    = 1e-2;
par.w      = par.w_star + eps_bif^2;

% ---------------- normalized uniform-band graph ----------------
% W_rho(x,y) = (1/(2*rho)) 1_{d_T(x,y) <= rho}
M = floor(rho*N);

if M < 1 || 2*M >= N
    error('Require 1 <= floor(rho*N) < N/2.');
end

shifts = [-M:-1, 1:M];

[I,S] = ndgrid((1:N)', shifts);
J = mod(I - 1 + S, N) + 1;

% There are 2M neighbors per row. Hence:
% (1/N) * rowSum(A) = (1/N)*(2M)*(N/(2M)) = 1.
edge_weight = N/(2*M);
A = sparse(I(:), J(:), edge_weight, N, N);

par.A = A;

%fprintf('(1/N) row sums: min = %.16g, max = %.16g\n',min(sum(A,2))/N, max(sum(A,2))/N);

% ---------------- reflection-symmetric initial data ----------------
% Grid: x_i = (i-1)/N.
% Reflection x -> 1-x maps MATLAB indices
% [1,2,...,N] -> [1,N,N-1,...,2].
seed = 21;
rng(seed);

eps_amp = 1e-2;

reflection_index = [1, N:-1:2];

z_n = randn(N,1);
z_c = randn(N,1);

z_n = 0.5 * (z_n + z_n(reflection_index));
z_c = 0.5 * (z_c + z_c(reflection_index));

n_init = par.n0 + eps_amp*z_n;

c_star = par.alpha*par.n0/par.beta;
c_init = c_star + eps_amp*z_c;

y0 = [n_init; c_init];

% ---------------- solve ----------------
T_final = 5000;
opts = odeset( ...
    'RelTol',1e-8, ...
    'AbsTol',1e-10, ...
    'NonNegative',1:2*N);

[t,y] = ode15s( ...
    @(t,y) rhs_uniform_band_graph(t,y,par), ...
    [0,T_final], y0, opts);

% Extract time histories
n_hist = y(:, 1:N);
c_hist = y(:, N+1:2*N);

% Manually select nodes
selected_nodes = [1, 6, 9, 15, 17];

% Check that all selected nodes are valid
if any(selected_nodes < 1) || any(selected_nodes > N)
    error('All selected nodes must be between 1 and N.');
end

% ---------------- t versus n_i(t) ----------------
figure;
plot(t, n_hist(:,selected_nodes), 'LineWidth', 2);

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
ylim([5.5, 10.5]);

legend(legend_labels, ...
    'Interpreter','latex', ...
    'Location','eastoutside', ...
    'FontSize',14);


% ---------------- t versus c_i(t) ----------------
figure;
plot(t, c_hist(:,selected_nodes), 'LineWidth', 2);

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

ylim([13, 19]);

legend(legend_labels, ...
    'Interpreter','latex', ...
    'Location','eastoutside', ...
    'FontSize',14);

n_final = y(end,1:N).';
c_final = y(end,N+1:2*N).';

x = (0:N-1)'/N;

% Final snapshots
n_snapshot = y(end, 1:N).';
c_snapshot = y(end, N+1:2*N).';

% ---------------- n scatter plot ----------------
figure;
plot(1:N, n_snapshot, '-o', 'LineWidth', 2, 'MarkerSize', 8, Color = "#FF5F05");

hold on;
yline(par.n0, '--k', 'LineWidth', 4);
hold off;

range_1 = 1.3*max(abs(n_snapshot - par.n0));
xlim([1, N])
ylim([par.n0 - range_1, par.n0 + range_1]);
xlabel('Node index $i$', ...
    'Interpreter','latex', 'FontSize',18);
ylabel('$n_i$', ...
    'Interpreter','latex', 'FontSize',18);
title(sprintf( ...
    'Snapshot of $n_i$ at $t=%g$ (Number of nodes $N=%d$)', ...
    T_final, N), ...
    'Interpreter','latex', 'FontSize',20);

grid on;
box on;


% ---------------- c scatter plot ----------------
figure;
plot(1:N, c_snapshot, '-o', 'LineWidth', 2, 'MarkerSize', 8, Color = "#FF5F05");

hold on;
yline(c_star, '--k', 'LineWidth', 4);
hold off;

range_2 = 1.3*max(abs(c_snapshot - par.c0));
xlim([1, N]);
ylim([par.c0 - range_2, par.c0 + range_2]);
xlabel('Node index $i$', ...
    'Interpreter','latex', 'FontSize',18);
ylabel('$c_i$', ...
    'Interpreter','latex', 'FontSize',18);
title(sprintf( ...
    'Snapshot of $c_i$ at $t=%g$ (Number of nodes $N=%d$)', ...
    T_final, N), ...
    'Interpreter','latex', 'FontSize',20);

grid on;
box on;
