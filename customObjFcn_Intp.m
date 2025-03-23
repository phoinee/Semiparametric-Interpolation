function loss = customObjFcn_Intp(params, X, Y, Sigma)
    sig_l = params(1);
    sig_f = params(2);
    K = customKernel(X, X, sig_l, sig_f) + Sigma^2*eye(length(X));
    L = chol(K, 'lower');
    alpha = L' \ (L \ Y);
    loss = 0.5 * (Y' * alpha) + sum(log(diag(L)));
end