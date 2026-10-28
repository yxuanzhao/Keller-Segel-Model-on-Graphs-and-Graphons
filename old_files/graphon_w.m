clc;

%% ---------------- parameters ----------------
N      = 128;
rho    = 0.10;
eta_n  = 0.05;
eta_c  = 0.04;
w_star = 0.3984580819264639;
w_star_2 = 0.43749999999999994;
eps      = 1e-2;
%w      = p.w_star + (p.eps)^2;
alpha  = 0.2;
beta   = 0.1;
r      = 0.1;
S      = 8.0;

% initial values
k0       = 7;
n0       = 8.0;
c0       = alpha*n0/beta;
delta_n  = 1.0;
delta_c  = 0.33636183;

%% ---------------- grid and kernel ----------------
x = (0:N-1)'/N; dx = 1/N;
T = 10000.0; dt = 1e-1;
nsteps = round(T/dt);
t = linspace(0, T, nsteps);

mode = 'vectorized';  
k = [0:floor(N/2), -floor((N-1)/2):-1]';
Khat = mysinc(2*pi*rho*k);
mA = 1;

p = struct('N',N,'rho',rho,'eta_n',eta_n,'eta_c',eta_c, ...
           'eps',eps,'alpha',alpha,'beta',beta,'r',r, ...
           'S',S,'k0',k0,'n0',n0,'c0',c0, ...
           'delta_n',delta_n,'delta_c',delta_c,'mode',mode,'dt',dt);

w_list = linspace(0, 0.8, 20);
seeds = [42 73921 582 93457 1203 45821 9912 2743 86530 44321];
int_n_T = zeros(size(w_list));
int_c_T = zeros(size(w_list));

length_w = numel(w_list);
length_seeds = numel(seeds);

last_len = 0;

for i = 1:length_w
    w = w_list(i);
    p.('w') = w;
    temp_n = 0;
    temp_c = 0;
    RHSfun = @(nv,cv) RHS(nv, cv, Khat, mA, p);

    for j = 1:length_seeds
        seed = seeds(j);
        rng(seed, 'twister');
        z = 2*rand(size(x)) - 1;  
        n = n0 + eps*z;
        c = c0 + eps*z;

        for step = 1:nsteps
            [Fn1,Fc1] = RHSfun(n,c);
            n1 = n + 0.5*p.dt*Fn1;  c1 = c + 0.5*p.dt*Fc1;

            [Fn2,Fc2] = RHSfun(n1,c1);
            n2 = n + 0.5*p.dt*Fn2;  c2 = c + 0.5*p.dt*Fc2;

            [Fn3,Fc3] = RHSfun(n2,c2);
            n3 = n + p.dt*Fn3;      c3 = c + p.dt*Fc3;

            [Fn4,Fc4] = RHSfun(n3,c3);

            n = n + (p.dt/6)*(Fn1 + 2*Fn2 + 2*Fn3 + Fn4);
            c = c + (p.dt/6)*(Fc1 + 2*Fc2 + 2*Fc3 + Fc4);

            %fprintf('\r Progress %.1f%%', step/nsteps*100);
            %pause(0.01)
            current_step = (i-1)*length_seeds*nsteps + (j-1)*nsteps + step;
            percent = current_step /(length_w*length_seeds*nsteps) * 100;
            %percent = (step/nsteps)*(j/length_seeds)*(i/length_w)*100;
            msg = sprintf('Progress: %.1f%%', percent);

            if last_len > 0
                fprintf(repmat('\b', 1, last_len));
            end

            fprintf('%s', msg);
            last_len = length(msg);
        end

        temp_n = temp_n + sum(n)*dx;
        temp_c = temp_c + sum(c)*dx;
    end
    int_n_T(i) = temp_n/numel(seeds);
    int_c_T(i) = temp_c/numel(seeds);
end

figure; hold on; box on; grid on;
xline(w_star, 'LineStyle', ':', 'Color', 'black', 'LineWidth', 5.0);
xline(w_star_2, 'LineStyle', ':', 'Color', 'black', 'LineWidth', 5.0);
h1 = scatter(w_list, int_n_T, 100, [1, 0.3725, 0.0196], 'filled');
h2 = yline(n0, 'LineStyle', '--', 'Color', '#13294B', 'LineWidth', 5.0);
legend([h1, h2], {'Total $n$','$n_0$'}, 'Location', 'southwest', 'FontSize', 24, 'Interpreter','latex');
dist = max(abs(int_n_T - n0));
ylim([n0 - 1.5*dist n0 + 1.5*dist]);
xlabel('$w$', 'Interpreter','latex', 'FontSize',24);
ylabel('Density', 'Interpreter','latex', 'FontSize',24);
title(sprintf('$w$ vs total $n$'), 'Interpreter','latex', 'FontSize',24);


figure; hold on; box on; grid on;
xline(w_star, 'LineStyle', ':', 'Color', 'black', 'LineWidth', 5.0);
xline(w_star_2, 'LineStyle', ':', 'Color', 'black', 'LineWidth', 5.0);
h1 = scatter(w_list, int_c_T, 100, [1, 0.3725, 0.0196], 'filled');
h2 = yline(c0, 'LineStyle', '--', 'Color', '#13294B', 'LineWidth', 5.0);
legend([h1, h2], {'Total $c$','$c_0$'}, 'Location', 'southwest', 'FontSize', 24, 'Interpreter','latex');
dist = max(abs(int_c_T - c0));
ylim([c0 - 1.5*dist c0 + 1.5*dist]);
xlabel('$w$', 'Interpreter','latex', 'FontSize',24);
ylabel('Density', 'Interpreter','latex', 'FontSize',24);
title(sprintf('$w$ vs total $c$'), 'Interpreter','latex', 'FontSize',24);

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
        Q  = (Nj.*Ei - Ni.*Ej) ./ (Ei + Ej);     % N×N

        Qhat    = fft(Q, [], 2);
        Krow    = (ones(N,1) * Khat.');
        rowConv = ifft(Qhat .* Krow, [], 2);
        In      = real(diag(rowConv));
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

Fn = p.eta_n * In + p.r*nv.*(1 - nv/p.S);
end

%% ======== sinc ========
function y = mysinc(z)
y = ones(size(z));
mask = (z~=0);
y(mask) = sin(z(mask))./z(mask);
end
