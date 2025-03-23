function K = customKernel(X1, X2, sig_l, sig_f)
    n1 = size(X1, 1);
    n2 = size(X2, 1);
    K = zeros(n1, n2);
    for i = 1:n1
        for j = 1:n2
            diff = X1(i, :) - X2(j, :);
            K(i, j) = sig_f^2*exp(-0.5 * sum((diff ./ sig_l').^2));
        end
    end
end