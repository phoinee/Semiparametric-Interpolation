clear; clc;

% flag = 1: true basis, flag = 2: wrong basis
flag = 2;

% nonlinear mass-spring-damper system parameters
k1 = 1; k2 = 0.1; c1 = 1; c2 = -0.1; m = 1; 
pTrainGrid = 0.2; vTrainGrid = 0.2;
pTestGrid = 0.01; vTestGrid = 0.01;

% generate raw data
[pTrain, vTrain, aTrain] = dataGeneration(k1,k2,c1,c2,m,pTrainGrid,vTrainGrid);
[pTest, vTest, aTrue] = dataGeneration(k1,k2,c1,c2,m,pTestGrid,vTestGrid);

% generate training data
xTrain = [];
yTrain = [];
for ii = 1:length(pTrain)
    for jj = 1:length(vTrain)
        xTrain = [xTrain; pTrain(ii), vTrain(jj)];
        yTrain = [yTrain; aTrain(ii,jj)];
    end
end

% generating test data
xTest = [];
yTrue = [];
for ii = 1:length(pTest)
    for jj = 1:length(vTest)
        xTest = [xTest; pTest(ii), vTest(jj)];
        yTrue = [yTrue; aTrue(ii,jj)];
    end
end

% kernel interpolation
sig_l = 0.1;
sig_f = 1;
Sig = 0.001;

opts = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'interior-point');
init_param = [sig_l, sig_f];

[opt_params_intp, fval_intp] = fmincon(@(params) customObjFcn_Intp(params, xTrain, yTrain, Sig), init_param, [], [], [], [], 0.1*Sig*ones(size(init_param)), [], [], opts);
% hyperparameter optimization for kernel interpolation
KK_intp = customKernel(xTrain,xTrain,opt_params_intp(1),opt_params_intp(2)) + Sig^2*eye(length(xTrain));
% gram matrix of interpolation

yTestIntp = zeros(length(xTest),1);
for ii = 1:length(xTest)
    yTestIntp(ii) = customKernel(xTest(ii,:),xTrain,opt_params_intp(1),opt_params_intp(2))/KK_intp*yTrain;
end
% kernel interpolation results
yTestIntpMesh = zeros(length(pTest));
for ii = 1:length(pTest)
    for jj = 1:length(vTest)
        yTestIntpMesh(ii,jj) = customKernel([pTest(ii), vTest(jj)],xTrain,opt_params_intp(1),opt_params_intp(2))/KK_intp*yTrain;
    end
end
% kernel interpolation results continued

RMSEIntp = sqrt((yTestIntp - yTrue)'*(yTestIntp - yTrue)/length(yTestIntp));
% RMSE for kernel interpolation

% semiparametric kernel interpolation
[opt_params_semi, fval_semi] = fmincon(@(params) customObjFcn_Semi(params, xTrain, yTrain, Sig, flag), init_param, [], [], [], [], 0.1*Sig*ones(size(init_param)), [], [], opts);
% hyperparameter optimization for semiparametric interpolation

KK_semi = customKernel(xTrain,xTrain,opt_params_semi(1),opt_params_semi(2)) + Sig^2*eye(length(xTrain));
% gram matrix for semiparametric interpolation

if flag == 1
    Psi = xTrain;
else
    Psi = xTrain.^2;
end

tmp1 = Psi'/KK_semi*Psi;
bHat = tmp1\Psi'/KK_semi*yTrain;
% basis coefficients
aHat = KK_semi\(yTrain - Psi*bHat);
% RKHS coefficients

yTestSemi = zeros(length(xTest),1);
for ii = 1:length(xTest)
    yTestSemi(ii) = customKernel(xTest(ii,:),xTrain,opt_params_semi(1),opt_params_semi(2))*aHat + xTest(ii,:)*bHat;
end
% semiparametric interpolation results
yTestSemiMesh = zeros(length(pTest));
for ii = 1:length(pTest)
    for jj = 1:length(vTest)
        yTestSemiMesh(ii,jj) = customKernel([pTest(ii), vTest(jj)],xTrain,opt_params_semi(1),opt_params_semi(2))*aHat + [pTest(ii), vTest(jj)]*bHat;
    end
end
% semiparametric interpolation results continued

RMSESemi = sqrt((yTestSemi - yTrue)'*(yTestSemi - yTrue)/length(yTestIntp));
% RMSE for semiparametric kernel interpolation

% Least squares approach
bLin = (Psi'*Psi)\Psi'*yTrain;
% Least squares coefficients
yTestLin = zeros(length(xTest),1);
for ii = 1:length(xTest)
    yTestLin(ii) = xTest(ii,:)*bLin;
end
yTestLinMesh = zeros(length(pTest));
for ii = 1:length(pTest)
    for jj = 1:length(vTest)
        yTestLinMesh(ii,jj) = [pTest(ii), vTest(jj)]*bLin;
    end
end
% Least squares results

RMSELin = sqrt((yTestLin - yTrue)'*(yTestLin - yTrue)/length(yTestIntp));
% RMSE of least squares approach

error_intp_mesh = yTestIntpMesh - aTrue;
error_semi_mesh = yTestSemiMesh - aTrue;

% mesh plot
% figure(1);
% subplot(3,1,1)
% mesh(pTest,vTest,aTrue,'EdgeColor',"green");
% hold on;
% mesh(pTest,vTest,yTestIntpMesh,'EdgeColor',"cyan");
% axis([-0.5 0.5 -0.5 0.5 -0.5 0.5])
% legend('true', 'KI',fontsize=40)
% subplot(3,1,2)
% mesh(pTest,vTest,aTrue,'EdgeColor',"green");
% hold on;
% mesh(pTest,vTest,yTestSemiMesh,'EdgeColor',"red");
% axis([-0.5 0.5 -0.5 0.5 -0.5 0.5])
% legend('true', 'SKI',fontsize=40)
% subplot(3,1,3)
% mesh(pTest,vTest,aTrue,'EdgeColor',"green");
% hold on;
% mesh(pTest,vTest,yTestLinMesh,'EdgeColor',"blue");
% axis([-0.5 0.5 -0.5 0.5 -0.5 0.5])
% legend('true', 'LS',fontsize=40)


figure(2);
mesh(pTest,vTest,abs(error_semi_mesh),'EdgeColor',"red","LineWidth",1.5);hold on;
mesh(pTest,vTest,abs(error_intp_mesh),'EdgeColor',"cyan","LineWidth",1.5);
legend('SKI error', 'KI error',fontsize=40)
xlabel('$p$','Interpreter','latex')
ylabel('$v$','Interpreter','latex')
zlabel('absolute error')
set(gca,"FontSize",40)
