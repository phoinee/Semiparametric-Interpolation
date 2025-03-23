# Semiparametric-Interpolation

main_fixed_hyperparameters.m file draws an error mesh plot for semiparametric kernel interpolation (SKI), kernel interpolation (KI) and parametrized least squares (LS) from fixed Gaussian kernel.
main_optimized_hyperparameters.m file draws an error mesh plot for semiparametric kernel interpolation (SKI) and kernel interpolation (KI) from optimized Gaussian kernel.
In each case, flag == 1 (line 3, 4) denotes when the correct basis functions (psi_1 = p, psi_2 = v) are used and flag == 2 denotes when the wrong basis functions (psi_1 = p^2, psi_2 = v^2) are used.
