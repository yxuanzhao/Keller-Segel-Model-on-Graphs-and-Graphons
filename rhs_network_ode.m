function dydt = rhs_network_ode(~, y, par)
    % y = [n_1..n_N, c_1..c_N]^T

    N = par.N;
    A = par.A;

    n = y(1:N);
    c = y(N+1:2*N);

    % ---------- dn/dt ----------
    % transport term:
    % (eta_n/N) * sum_j A_ij * [ (exp(w c_i) n_j - exp(w c_j) n_i)/(exp(w c_i)+exp(w c_j)) ]

    E = exp(par.w * c);          % N x 1, E_i = exp(w c_i)

    dn = zeros(N,1);
    for i = 1:N
        idx = find(A(i,:) ~= 0); % neighbors
        if ~isempty(idx)
            Ei = E(i);
            ni = n(i);

            Ej = E(idx);
            nj = n(idx);

            frac = (Ei .* nj - Ej .* ni) ./ (Ei + Ej);  % vector over neighbors j
            dn(i) = dn(i) + (par.eta_n) * sum(frac);
        end
    end

    % growth term (default: node-wise logistic)
    dn = dn + par.rn * n .* (1 - n / par.n0);
    
    deg = sum(A,2); % degree vector
    dc = (par.eta_c) * (A*c - deg .* c) + par.alpha * n - par.beta * c;

    dydt = [dn; dc];
end