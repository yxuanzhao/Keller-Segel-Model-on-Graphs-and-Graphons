function new_reaction_graphon()

%clear;
clc;

%% ---------------- parameters ----------------
p.N      = 128;
p.T      = 400000.0;
p.dt     = 1e-1;
p.rho    = 0.10;
p.eta_n  = 0.05;
p.eta_c  = 0.04;
%p.w      = 3.45;
p.w_star = 0.3984580819264639;
p.w_star_star = 0.43749999999999994;
p.eps      = 1e-2;
p.w      = p.w_star + (p.eps)^2;
%p.w      = 2*p.w_star;
%p.w      = 7;
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
p.theta = 2/p.n0;

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
if_random = true;


if if_random
    random_seed = 42;
    rng(random_seed,'twister');
    %z = 2*rand(size(x)) - 1;  
    %z(end) = z(1);
    N = length(x);
    z = zeros(size(x));
    half = floor(N/2);   % 前半部分（不含中点）
    z(1:half+1) = 2*rand(1, half+1) - 1;
    for i = half+2:N
        z(i) = z(N - i + 2);
    end
    %z(end) = z(1);
    n = p.n0 + p.eps*z;
    c = p.c0 + p.eps*z;
else
    n = p.n0 + p.eps*p.delta_n*cos(2*pi*p.k0*x);
    c = p.c0 + p.eps*p.delta_c*cos(2*pi*p.k0*x);
end

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

fprintf('Progress: ');
last_len = 0;

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

    n_hist(step+1,:) = n.';
    c_hist(step+1,:) = c.';

    %fprintf('\r Progress %.1f%%', step/nsteps*100);
    %pause(0.01)
    percent = step / nsteps * 100;
    msg = sprintf('%.1f%%', percent);

    if last_len > 0
        fprintf(repmat('\b', 1, last_len));
    end

    fprintf('%s', msg);
    last_len = length(msg);
end

%% ---------------- heatmap ----------------

%{
% n heatmap
figure('Name','n(x,t) heatmap');
imagesc(x, tvec, n_hist);
set(gca,'YDir','normal');
colorbar;  
xlabel('Position $x$', 'Interpreter', 'latex', 'FontSize',18); 
ylabel('Time $t$', 'Interpreter', 'latex', 'FontSize',18);
title(sprintf('Heatmap of $n(x,t)$'), 'Interpreter','latex', 'FontSize',18);

% c heatmap
figure('Name','c(x,t) heatmap');
imagesc(x, tvec, c_hist);
set(gca,'YDir','normal');
colorbar;  
xlabel('Position $x$', 'Interpreter', 'latex', 'FontSize',18); 
ylabel('Time $t$', 'Interpreter', 'latex', 'FontSize',18);
title(sprintf('Heatmap of $c(x,t)$'), 'Interpreter','latex', 'FontSize',18);
%}

%{
figure;
plot(x, n, 'LineWidth', 5, Color = "#FF5F05");
yline(p.n0, '--black', 'LineWidth', 5);
range = max(abs(n - p.n0));
ylim([p.n0 - 1.25*range, p.n0 + 1.25*range]);
xlabel('Position $x$', 'Interpreter','latex', 'FontSize',24); 
ylabel('$n$', 'Interpreter','latex', 'FontSize',24);
%title('Snapshot of $n(t,x)$ at $t=50000$','Interpreter','latex', 'FontSize',24); 
title(sprintf('Snapshot of $n(t,x)$ at $t=%.0f$', p.T),'Interpreter','latex', 'FontSize',24);
grid on;

figure;
plot(x, c, 'LineWidth', 5, Color = "#FF5F05");
yline(p.c0, '--black', 'LineWidth', 5);
range = max(abs(c - p.c0));
ylim([p.c0 - 1.25*range, p.c0 + 1.25*range]);
xlabel('Position $x$', 'Interpreter','latex', 'FontSize',24); 
ylabel('$c$', 'Interpreter','latex', 'FontSize',24);
%title('Snapshot of $n(t,x)$ at $t=50000$','Interpreter','latex', 'FontSize',24); 
title(sprintf('Snapshot of $c(t,x)$ at $t=%.0f$', p.T),'Interpreter','latex', 'FontSize',24);
grid on;
%}

figure;
plot(t, int_n, 'LineWidth', 5, Color = "#FF5F05");
yline(p.n0, '--black', 'LineWidth', 5);
range = max(abs(int_n - p.n0));
ylim([p.n0 - 1.25*range, p.n0 + 1.25*range]);
xlabel('Time $t$', 'Interpreter','latex', 'FontSize',24); 
ylabel('Total of $n$', 'Interpreter','latex', 'FontSize',24);
title('Total number of $n$ over time','Interpreter','latex', 'FontSize',24); 
grid on;

figure;
plot(t, int_c, 'LineWidth', 5, Color = "#FF5F05");
yline(p.c0, '--black', 'LineWidth', 5);
range = max(abs(int_c - p.c0));
ylim([p.c0 - 1.25*range, p.c0 + 1.25*range]);
xlabel('Time $t$', 'Interpreter','latex', 'FontSize',24); 
ylabel('Total of $c$', 'Interpreter','latex', 'FontSize',24);
title('Total number of $c$ over time','Interpreter','latex', 'FontSize',24); 
grid on;

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

Fn = p.eta_n * In + p.r*exp(p.theta * (p.S - nv)).*nv.*(1 - nv/p.S);
end

%% ======== sinc ========
function y = mysinc(z)
y = ones(size(z));
mask = (z~=0);
y(mask) = sin(z(mask))./z(mask);
end
