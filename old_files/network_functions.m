function dydt = network_functions(~, y, par)
    % y = [n_1..n_N, c_1..c_N]^T

    N = par.N;
    A = par.A;
    eta_n = par.eta_n;
    eta_c = par.eta_c;
    alpha = par.alpha;
    beta = par.beta;

    n = y(1:N);
    c = y(N+1:2*N);

    % ---------- dn/dt ----------
    % transport term:
    % (eta_n/N) * sum_j A_ij * [ (exp(w c_i) n_j - exp(w c_j) n_i)/(exp(w c_i)+exp(w c_j)) ]

    E = exp(par.w * c);          % N x 1, E_i = exp(w c_i)

    dn = zeros(N,1);
    dc = zeros(N,1);

    for i = 1:N
        dn(i) = 0;
        dc(i) = 0;
        for j = 1:N
            dn(i) = dn(i) + eta_n*A(i, j)*(E(i)*n(j) - E(j)*n(i))/(E(i) + E(j));
            dc(i) = dc(i) + eta_c*A(i, j)*(c(j) - c(i))/2;
        end
    end

    % growth term (default: node-wise logistic)
    dn = dn + par.rn * n .* (1 - n / par.n0);
    dc = dc + alpha*n - beta*c;

    dydt = [dn; dc];
end
