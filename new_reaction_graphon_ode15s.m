function new_reaction_graphon_ode15s()

clc;

%% ---------------- parameters ----------------
p.N      = 128;
p.T      = 500000.0;

p.rho    = 0.10;
p.eta_n  = 0.05;
p.eta_c  = 0.04;

p.w_star = 0.3984580819264639;
p.w_star_star = 0.43749999999999994;
p.eps    = 1e-2;
p.w      = p.w_star + (p.eps)^2;

p.alpha  = 0.2;
p.beta   = 0.1;
p.r      = 0.1;
p.S      = 8.0;

% initial values
p.k0       = 7;
p.n0       = 8.0;
p.c0       = 16.0;
p.delta_n  = 1.0;
p.delta_c  = 0.33636183;
p.theta    = 1.1/p.n0;
%p.theta    = 0;

% options
p.mode        = 'vectorized';   % 'vectorized' or 'rowloop'
p.use_exact_A = true;

%% ---------------- grid and kernel ----------------
N  = p.N;
x  = (0:N-1)'/N;
dx = 1/N;

if p.use_exact_A
    k = [0:floor(N/2), -floor((N-1)/2):-1]';
    Khat = mysinc(2*pi*p.rho*k);
    mA = 1;
else
    dwrap = @(s) min(mod(abs(s),1), 1 - mod(abs(s),1));
    K = double(dwrap((0:N-1)'/N) <= p.rho);
    K = K / (sum(K)*dx);
    Khat = fft(K)*dx;
    mA = real(Khat(1));
end

%% ---------------- initial values ----------------
if_random = true;

if if_random
    random_seed = 42;
    rng(random_seed,'twister');

    z = zeros(size(x));
    half = floor(N/2);
    z(1:half+1) = 2*rand(1, half+1) - 1;

    for i = half+2:N
        z(i) = z(N - i + 2);
    end

    n_init = p.n0 + p.eps*z;
    c_init = p.c0 + p.eps*z;
else
    n_init = p.n0 + p.eps*p.delta_n*cos(2*pi*p.k0*x);
    c_init = p.c0 + p.eps*p.delta_c*cos(2*pi*p.k0*x);
end

y0 = [n_init; c_init];

%% ---------------- ode15s adaptive solver ----------------

num_output = 4001;
t_eval = linspace(0, p.T, num_output);

opts = odeset( ...
    'RelTol', 1e-8, ...
    'AbsTol', 1e-10, ...
    'NonNegative', 1:2*N, ...
    'Stats', 'on');

fprintf('Solving with ode15s...\n');

[t, y] = ode15s(@(t,y) RHS_vec(t, y, Khat, mA, p), t_eval, y0, opts);

fprintf('Finished.\n');

%% ---------------- extract solution ----------------
n_hist = y(:, 1:N);
c_hist = y(:, N+1:2*N);

n_final = n_hist(end,:).';
c_final = c_hist(end,:).';

int_n = sum(n_hist, 2) * dx;
int_c = sum(c_hist, 2) * dx;

fprintf('min(n) = %.16g, max(n) = %.16g\n', min(n_final), max(n_final));
fprintf('min(c) = %.16g, max(c) = %.16g\n', min(c_final), max(c_final));
fprintf('final total n = %.16g\n', int_n(end));
fprintf('final total c = %.16g\n', int_c(end));

%% ---------------- weakly nonlinear approximation ----------------
p_n = 0.53268804;
p_c = 0.34732833;
q_n = 1.0;
q_c = 1.34544731;
k_star = 7;
%t_0n = 0.0625;
%t_0c = 0.125;
%t_2n = 0.08540532589869312;
%t_2c = 0.1243822671618732;
t_0n = 0.00625;
t_0c = 0.0125;
%t_2n = 0.03417174536693547;
%t_2c = 0.0497669099308795;
t_2n = 0.24972113877705873;
t_2c = 0.3636878739410164;
kappa = 0.17433619427681338;
%gamma = -0.0017076092662578477;
gamma = -0.0023774828528576072;
A_abs = sqrt(-kappa/gamma);

% determine sign from initial projection
phi = cos(2*pi*k_star*x);
A0 = sum((p_n*(n_init - p.n0) + p_c*(c_init - p.c0)) .* phi) * dx;
A_final = sign(A0) * A_abs;
if A_final == 0
    A_final = A_abs;
end
n_wna = p.n0 + p.eps*A_final*q_n*cos(2*pi*k_star*x) + p.eps^2*A_final^2*(t_0n + t_2n*cos(4*pi*k_star*x));
c_wna = p.c0 + p.eps*A_final*q_c*cos(2*pi*k_star*x) + p.eps^2*A_final^2*(t_0c + t_2c*cos(4*pi*k_star*x));

%% ---------------- plots ----------------

figure;
plot(t, int_n, 'LineWidth', 5, Color = "#FF5F05");
yline(p.n0, '--black', 'LineWidth', 5);
range = max(abs(int_n - p.n0));
if range == 0
    range = 1e-8;
end
ylim([p.n0 - 1.25*range, p.n0 + 1.25*range]);
xlabel('Time $t$', 'Interpreter','latex', 'FontSize',24);
ylabel('Total of $n$', 'Interpreter','latex', 'FontSize',24);
title('Total number of $n$ over time','Interpreter','latex', 'FontSize',24);
grid on;

figure;
plot(t, int_c, 'LineWidth', 5, Color = "#FF5F05");
yline(p.c0, '--black', 'LineWidth', 5);
range = max(abs(int_c - p.c0));
if range == 0
    range = 1e-8;
end
ylim([p.c0 - 1.25*range, p.c0 + 1.25*range]);
xlabel('Time $t$', 'Interpreter','latex', 'FontSize',24);
ylabel('Total of $c$', 'Interpreter','latex', 'FontSize',24);
title('Total number of $c$ over time','Interpreter','latex', 'FontSize',24);
grid on;

figure;
plot(x, n_final, 'LineWidth', 5, Color = "#FF5F05");
hold on;
plot(x, n_wna, ':', 'LineWidth', 5, Color = "#13294B");
yline(p.n0, '--black', 'LineWidth', 3);
range = max(abs([n_final - p.n0; n_wna - p.n0]));
if range == 0
    range = 1e-8;
end
ylim([p.n0 - 1.25*range, p.n0 + 1.25*range]);
xlabel('Position $x$', 'Interpreter','latex', 'FontSize',24);
ylabel('$n$', 'Interpreter','latex', 'FontSize',24);
title(sprintf('Snapshot of $n(t,x)$ at $t=%.0f$', p.T), ...
    'Interpreter','latex', 'FontSize',24);
%legend({'Numerical solution','WNA approximation','$n_0$'}, ...
%    'Interpreter','latex', ...
%    'Location','best', ...
%    'FontSize',18);
grid on;
hold off;

figure;
plot(x, c_final, 'LineWidth', 5, Color = "#FF5F05");
hold on;
plot(x, c_wna, ':', 'LineWidth', 5, Color = "#13294B");
yline(p.c0, '--black', 'LineWidth', 3);
range = max(abs([c_final - p.c0; c_wna - p.c0]));
if range == 0
    range = 1e-8;
end
ylim([p.c0 - 1.25*range, p.c0 + 1.25*range]);
xlabel('Position $x$', 'Interpreter','latex', 'FontSize',24);
ylabel('$c$', 'Interpreter','latex', 'FontSize',24);
title(sprintf('Snapshot of $c(t,x)$ at $t=%.0f$', p.T), ...
    'Interpreter','latex', 'FontSize',24);
%legend({'Numerical solution','WNA approximation','$c_0$'}, ...
%    'Interpreter','latex', ...
%    'Location','best', ...
%    'FontSize',18);
grid on;
hold off;

%% ---------------- heatmap ----------------

skip = 20;
idx = 1:skip:length(t);

figure('Name','n(x,t) heatmap');
imagesc(x, t, n_hist);
set(gca,'YDir','normal');
colorbar;
xlabel('Position $x$','Interpreter','latex','FontSize',24);
ylabel('Time $t$','Interpreter','latex','FontSize',24);
title('Heatmap of $n(x,t)$ ','Interpreter','latex','FontSize',24);

figure('Name','c(x,t) heatmap');
imagesc(x, t, c_hist);
set(gca,'YDir','normal');
colorbar;
xlabel('Position $x$','Interpreter','latex','FontSize',24);
ylabel('Time $t$','Interpreter','latex','FontSize',24);
title('Heatmap of $c(x,t)$','Interpreter','latex','FontSize',24);

end


%% ======== vector RHS for ode15s ========
function dy = RHS_vec(~, y, Khat, mA, p)

N = p.N;

nv = y(1:N);
cv = y(N+1:2*N);

[Fn, Fc] = RHS(nv, cv, Khat, mA, p);

dy = [Fn; Fc];

end


%% ======== RHS ========
function [Fn, Fc] = RHS(nv, cv, Khat, mA, p)

N = numel(nv);

% ---- c ----
convAc = real(ifft( fft(cv) .* Khat ));
Lc     = p.eta_c * (convAc - mA*cv);
Fc     = Lc + p.alpha*nv - p.beta*cv;

% ---- n ----
E  = exp(p.w * cv);

switch p.mode
    case 'vectorized'
        Ei = E * ones(1,N);
        Ej = ones(N,1) * E.';
        Ni = nv * ones(1,N);
        Nj = ones(N,1) * nv.';

        Q  = (Nj.*Ei - Ni.*Ej) ./ (Ei + Ej);

        Qhat    = fft(Q, [], 2);
        Krow    = ones(N,1) * Khat.';
        rowConv = ifft(Qhat .* Krow, [], 2);

        In = real(diag(rowConv));

    case 'rowloop'
        In = zeros(N,1);
        for i = 1:N
            Ei = E(i);
            qi = (nv.*Ei - nv(i).*E) ./ (Ei + E);
            conv_row = ifft( fft(qi) .* Khat );
            In(i) = real(conv_row(i));
        end

    otherwise
        error('p.mode must be ''vectorized'' or ''rowloop''');
end

reaction = p.r * exp(p.theta * (p.S - nv)) .* nv .* (1 - nv/p.S);

Fn = p.eta_n * In + reaction;

end


%% ======== sinc ========
function y = mysinc(z)

y = ones(size(z));
mask = (z ~= 0);
y(mask) = sin(z(mask))./z(mask);

end