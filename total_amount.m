function graphon()

%% ---------------- parameters ----------------
p.N      = 256;
p.T      = 10000.0;
p.dt     = 5e-2;
p.rho    = 0.10;
p.eta_n  = 0.03;
p.eta_c  = 0.05;
p.w      = 4.75;
%p.w      = 4.68748
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

%% ---------------- RK4 ----------------
nsteps = round(p.T/p.dt);
tvec   = (0:nsteps).' * p.dt;
t = linspace(0, p.T, nsteps);
int_n = zeros(nsteps,1);
int_c = zeros(nsteps,1);

n_hist = zeros(nsteps+1, N);
c_hist = zeros(nsteps+1, N);
n_hist(1,:) = n.';
c_hist(1,:) = c.';

RHSfun = @(nv,cv) RHS(nv, cv, Khat, mA, p);

for step = 1:nsteps
    int_n(step) = sum(n) * dx;
    int_c(step) = sum(c) * dx;


    [Fn1,Fc1] = RHSfun(n,c);
    n1 = n + 0.5*p.dt*Fn1;  c1 = c + 0.5*p.dt*Fc1;

    [Fn2,Fc2] = RHSfun(n1,c1);
    n2 = n + 0.5*p.dt*Fn2;  c2 = c + 0.5*p.dt*Fc2;

    [Fn3,Fc3] = RHSfun(n2,c2);
    n3 = n + p.dt*Fn3;      c3 = c + p.dt*Fc3;

    [Fn4,Fc4] = RHSfun(n3,c3);

    n = n + (p.dt/6)*(Fn1 + 2*Fn2 + 2*Fn3 + Fn4);
    c = c + (p.dt/6)*(Fc1 + 2*Fc2 + 2*Fc3 + Fc4);
end


figure;
plot(t, int_n, 'LineWidth', 1.6, Color = "blue");
yline(p.n0, '--black', 'LineWidth', 1.5);
%xlabel('t'); ylabel('Total of u');
title('Total number of $n$ over time','Interpreter','latex'); 
grid on;

figure;
plot(t, int_c, 'LineWidth', 1.6, Color = "blue");
yline(p.c0, '--black', 'LineWidth', 1.5);
%xlabel('t'); ylabel('Total of u');
title('Total number of $c$ over time','Interpreter','latex'); 
grid on;


end


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
