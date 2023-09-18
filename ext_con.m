function [c, ceq] = ext_con(x)
    c   = [];
    ceq = norm(x, 2) - numel(x);
end