function loss = customObjFcn_Semi(params, X, Y, Sigma, flag)
    sig_l = params(1);
    sig_f = params(2);
    K = customKernel(X, X, sig_l, sig_f) + Sigma^2*eye(length(X));
    L = chol(K, 'lower');
    if flag == 1 % correct hyperparameters
        zz = Y - X/(X'/K*X)*X'/K*Y; % zz = y - Psi b
    else
        zz = Y - X.^2/(X.^2'/K*X.^2)*X.^2'/K*Y; % zz = y - Psi b
    end
    loss = 0.5 * zz'/K*zz + sum(log(diag(L)));
end
