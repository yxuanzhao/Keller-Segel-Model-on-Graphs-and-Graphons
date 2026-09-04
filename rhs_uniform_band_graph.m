function dydt = rhs_uniform_band_graph(~, y, par)

N = par.N;
A = par.A;

n = y(1:N);
c = y(N+1:2*N);

E = exp(par.w*c);

% ---------------- population transport ----------------
dn_transport = zeros(N,1);

for i = 1:N
    idx = find(A(i,:));
    idx = idx(:);

    aij = full(A(i,idx)).';

    flux_ji = ...
        (E(i).*n(idx) - E(idx).*n(i)) ./ ...
        (E(i) + E(idx));

    dn_transport(i) = ...
        (par.eta_n/N) * sum(aij .* flux_ji);
end

reaction = par.rn*n.*(1 - n/par.n0);
dn = dn_transport + reaction;

% ---------------- chemical diffusion ----------------
deg = full(sum(A,2));

dc = (par.eta_c/(2*N)) * (A*c - deg.*c) ...
     + par.alpha*n ...
     - par.beta*c;

dydt = [dn; dc];

end
