X    = fmri_gen_pseudodat_ycgosu(100, 10);
bet = 0.04;


W0 = corr(X);
W0(logical(eye(size(W0)))) = 0;
fun = @(x)(-x'*W0 *x)/2;

a0         = X(1, :)';
at         = [];
for i = 1:10000
    if i == 1
        at = tanh(bet * W0 * a0 + randn(size(a0)));
    else 
        at = tanh(bet * W0 * at + randn(size(a0)));
    end
    Ats(:,i) = at;
end

