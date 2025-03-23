function loss = customObjFcn_Semi(params, X, Y,Sigma)
    sig_l = params(1);
    sig_f = params(2);
    K = customKernel(X, X, sig_l, sig_f) + Sigma^2*eye(length(X));
    L = chol(K, 'lower');
    zz = Y - X/(X'/K*X)*X'/K*Y; % zz = y - Psi b
    loss = 0.5 * zz'/K*zz + sum(log(diag(L)));
end