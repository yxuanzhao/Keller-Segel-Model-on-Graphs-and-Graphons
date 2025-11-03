function graphon()
%% ---------------- parameters ----------------
p.N      = 256;
p.T      = 20000.0;
p.dt     = 1e-1;
p.rho    = 0.10;
p.eta_n  = 0.03;
p.eta_c  = 0.05;
p.w      = 4.75;
%p.w      = 4.9;
p.w_star = 4.68748;
p.alpha  = 0.05;
p.beta   = 0.15;
p.r      = 0.1;
p.S      = 6.0;

% initial values
p.k0       = 6;
p.n0       = 6.0;
p.c0       = 2.0;
p.eps      = 1e-2;
p.delta_n  = 1.0;
p.delta_c  = 0.240621;

% options
p.mode        = 'vectorized';   % 'vectorized' or 'rowloop'
p.use_exact_A = true;
p.do_bench    = true;

% WNL coefficients
wnl.qn   = 1.0;
wnl.qc   = 0.240621;
wnl.t0n  = -0.18111560267455967;
wnl.t2n  = -0.14823527265018024;
wnl.t0c  = -0.060371867558186555;
wnl.t2c  = -0.03826549953043763;
wnl.kappa  = 0.5332785341585228;
wnl.gamma  = -0.6386021384027721;

%% ---------------- grid and kernel ----------------
N  = p.N;  x = (0:N-1)'/N;  dx = 1/N;

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
n = p.n0 + p.eps*p.delta_n*cos(2*pi*p.k0*x);
c = p.c0 + p.eps*p.delta_c*cos(2*pi*p.k0*x);

%% ---------------- ODE setup ----------------
eps2    = p.eps^2;
tilde_w = (p.w - p.w_star)/eps2;
Kcap    = wnl.kappa * tilde_w;
f_a     = @(a) eps2 * (Kcap*a + wnl.gamma*a^3);   % da/dt
a       = 1;                               % a(0)

nsteps = round(p.T/p.dt);
RHSfun = @(nv,cv) RHS(nv, cv, Khat, mA, p);

if p.do_bench
    [~,~] = RHSfun(n,c);
    iters = 5; t0 = tic;
    for ii=1:iters, [~,~] = RHSfun(n,c); end
    t_rhs = toc(t0)/iters;
    fprintf('[Benchmark] single RHS: %.4f s (N=%d, %s, exactA=%d)\n', ...
        t_rhs, N, p.mode, p.use_exact_A);
    fprintf('  estimated steps(RK4×4RHS): %.4f s,  total steps: %d\n', 4*t_rhs, nsteps);
end

for step = 1:nsteps
    % ---- PDE: RK4 ----
    [Fn1,Fc1] = RHSfun(n,c);
    n1 = n + 0.5*p.dt*Fn1;  c1 = c + 0.5*p.dt*Fc1;

    [Fn2,Fc2] = RHSfun(n1,c1);
    n2 = n + 0.5*p.dt*Fn2;  c2 = c + 0.5*p.dt*Fc2;

    [Fn3,Fc3] = RHSfun(n2,c2);
    n3 = n + p.dt*Fn3;      c3 = c + p.dt*Fc3;

    [Fn4,Fc4] = RHSfun(n3,c3);
    n = n + (p.dt/6)*(Fn1 + 2*Fn2 + 2*Fn3 + Fn4);
    c = c + (p.dt/6)*(Fc1 + 2*Fc2 + 2*Fc3 + Fc4);

    % ---- amplitude: RK4 ----
    k1 = f_a(a);
    k2 = f_a(a + 0.5*p.dt*k1);
    k3 = f_a(a + 0.5*p.dt*k2);
    k4 = f_a(a + p.dt*k3);
    a  = a + (p.dt/6)*(k1 + 2*k2 + 2*k3 + k4);
end

%% ---------------- final WNL ----------------
cos1 = cos(2*pi*p.k0*x);    % N×1
cos2 = cos(4*pi*p.k0*x);    % N×1
n_appro = p.n0 + p.eps*a*wnl.qn .* cos1 ...
        + (p.eps^2)*(a^2) * ( wnl.t0n + wnl.t2n .* cos2 );

c_appro = p.c0 + p.eps*a*wnl.qc .* cos1 ...
        + (p.eps^2)*(a^2) * ( wnl.t0c + wnl.t2c .* cos2 );

err_n_Linf = norm(n - n_appro, inf);
err_c_Linf = norm(c - c_appro, inf);
fprintf('Final-time errors (L_inf):  ||n - nappro||=%.3e,  ||c - cappro||=%.3e\n', ...
        err_n_Linf, err_c_Linf);

% --- n: PDE vs WNL ---
figure('Name','n at final time');
plot(x, n, 'b-', 'LineWidth', 1.6); hold on;
plot(x, n_appro, 'r--', 'LineWidth', 1.6);
grid on; xlim([0 1]);
xlabel('x'); ylabel('n');
title(sprintf('$n(x, T=%.2f)$: PDE (blue) vs WNL (red dashed)', p.T),'Interpreter','latex');
legend({'PDE solution','WNL approx'}, 'Location','best','Interpreter','latex');

% --- c: PDE vs WNL ---
figure('Name','c at final time');
plot(x, c, 'b-', 'LineWidth', 1.6); hold on;
plot(x, c_appro, 'r--', 'LineWidth', 1.6);
grid on; xlim([0 1]);
xlabel('x'); ylabel('c');
title(sprintf('$c(x, T=%.2f)$: PDE (blue) vs WNL (red dashed)', p.T),'Interpreter','latex');
legend({'PDE solution','WNL approx'}, 'Location','best','Interpreter','latex');

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
