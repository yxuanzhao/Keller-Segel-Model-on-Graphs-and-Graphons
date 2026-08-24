clc;

%% ---------------- parameters ----------------
N      = 128;
rho    = 0.10;
eta_n  = 0.05;
eta_c  = 0.04;
w_star = 0.3984580819264639;
w_star_2 = 0.43749999999999994;
eps      = 1e-2;

alpha  = 0.2;
beta   = 0.1;
r      = 0.1;
S      = 8.0;

k0       = 7;
n0       = 8.0;
c0       = alpha*n0/beta;
delta_n  = 1.0;
delta_c  = 0.33636183;

%% ---------------- grid and kernel ----------------
x = (0:N-1)'/N;
dx = 1/N;

T = 50000.0;
num_output = 101;
t_eval = linspace(0, T, num_output);

mode = 'vectorized';

k = [0:floor(N/2), -floor((N-1)/2):-1]';
Khat = mysinc(2*pi*rho*k);
mA = 1;

p = struct('N',N,'rho',rho,'eta_n',eta_n,'eta_c',eta_c, ...
           'eps',eps,'alpha',alpha,'beta',beta,'r',r, ...
           'S',S,'k0',k0,'n0',n0,'c0',c0, ...
           'delta_n',delta_n,'delta_c',delta_c,'mode',mode);

w_list = linspace(0, 0.8, 30);
seeds = [42 73921 582 93457 1203 45821 9912 2743 86530 44321];

int_n_T = zeros(size(w_list));
int_c_T = zeros(size(w_list));

length_w = numel(w_list);
length_seeds = numel(seeds);

opts = odeset( ...
    'RelTol', 1e-8, ...
    'AbsTol', 1e-10, ...
    'NonNegative', 1:2*N);

last_len = 0;

for i = 1:length_w
    w = w_list(i);
    p.w = w;

    temp_n = 0;
    temp_c = 0;

    for j = 1:length_seeds
        seed = seeds(j);
        rng(seed, 'twister');

        z = 2*rand(size(x)) - 1;
        n_init = n0 + eps*z;
        c_init = c0 + eps*z;

        y0 = [n_init; c_init];

        [t, y] = ode15s(@(t,y) RHS_vec(t, y, Khat, mA, p), t_eval, y0, opts);

        n_final = y(end, 1:N).';
        c_final = y(end, N+1:2*N).';

        temp_n = temp_n + sum(n_final)*dx;
        temp_c = temp_c + sum(c_final)*dx;

        current_job = (i-1)*length_seeds + j;
        percent = current_job/(length_w*length_seeds)*100;
        msg = sprintf('Progress: %.1f%%', percent);

        if last_len > 0
            fprintf(repmat('\b', 1, last_len));
        end
        fprintf('%s', msg);
        last_len = length(msg);
    end

    int_n_T(i) = temp_n/length_seeds;
    int_c_T(i) = temp_c/length_seeds;
end

fprintf('\nFinished.\n');

%% ---------------- plot total n ----------------
figure; hold on; box on; grid on;
xline(w_star, 'LineStyle', ':', 'Color', 'black', 'LineWidth', 5.0);
xline(w_star_2, 'LineStyle', ':', 'Color', 'black', 'LineWidth', 5.0);
h1 = scatter(w_list, int_n_T, 100, [1, 0.3725, 0.0196], 'filled');
h2 = yline(n0, 'LineStyle', '--', 'Color', '#13294B', 'LineWidth', 5.0);
legend([h1, h2], {'Total $n$','$n_0$'}, ...
    'Location', 'southwest', 'FontSize', 24, 'Interpreter','latex');
dist = max(abs(int_n_T - n0));
if dist == 0
    dist = 1e-8;
end
ylim([n0 - 1.5*dist, n0 + 1.5*dist]);
xlabel('$\chi$', 'Interpreter','latex', 'FontSize',24);
ylabel('Population Size $\int_0^1 n(t,x)\,dx$', 'Interpreter','latex', 'FontSize',24);
title('$\chi$ vs total $n$ (logistic reaction)', 'Interpreter','latex', 'FontSize',24);

%% ---------------- plot total c ----------------
figure; hold on; box on; grid on;
xline(w_star, 'LineStyle', ':', 'Color', 'black', 'LineWidth', 5.0);
xline(w_star_2, 'LineStyle', ':', 'Color', 'black', 'LineWidth', 5.0);
h1 = scatter(w_list, int_c_T, 100, [1, 0.3725, 0.0196], 'filled');
h2 = yline(c0, 'LineStyle', '--', 'Color', '#13294B', 'LineWidth', 5.0);
legend([h1, h2], {'Total $c$','$c_0$'}, ...
    'Location', 'southwest', 'FontSize', 24, 'Interpreter','latex');
dist = max(abs(int_c_T - c0));
if dist == 0
    dist = 1e-8;
end
ylim([c0 - 1.5*dist, c0 + 1.5*dist]);
xlabel('$\chi$', 'Interpreter','latex', 'FontSize',24);
ylabel('Total Attractant $\int_0^1 c(t,x)\, dx$', 'Interpreter','latex', 'FontSize',24);
title('$\chi$ vs total $c$ (logistic reaction)', 'Interpreter','latex', 'FontSize',24);


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
convAc = real(ifft(fft(cv) .* Khat));
Lc     = p.eta_c * (convAc - mA*cv);
Fc     = Lc + p.alpha*nv - p.beta*cv;

% ---- n ----
E = exp(p.w * cv);

switch p.mode
    case 'vectorized'
        Ei = E * ones(1,N);
        Ej = ones(N,1) * E.';
        Ni = nv * ones(1,N);
        Nj = ones(N,1) * nv.';

        Q = (Nj.*Ei - Ni.*Ej) ./ (Ei + Ej);

        Qhat    = fft(Q, [], 2);
        Krow    = ones(N,1) * Khat.';
        rowConv = ifft(Qhat .* Krow, [], 2);

        In = real(diag(rowConv));

    case 'rowloop'
        In = zeros(N,1);
        for i = 1:N
            Ei = E(i);
            qi = (nv.*Ei - nv(i).*E) ./ (Ei + E);
            conv_row = ifft(fft(qi) .* Khat);
            In(i) = real(conv_row(i));
        end

    otherwise
        error('p.mode must be ''vectorized'' or ''rowloop''');
end

reaction = p.r * nv .* (1 - nv/p.S);

Fn = p.eta_n * In + reaction;

end


%% ======== sinc ========
function y = mysinc(z)

y = ones(size(z));
mask = (z ~= 0);
y(mask) = sin(z(mask))./z(mask);

end