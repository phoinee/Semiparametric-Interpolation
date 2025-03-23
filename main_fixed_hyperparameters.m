clear; clc;

% flag = 1: true basis, flag = 2: wrong basis
flag = 1;

% nonlinear mass-spring-damper system parameters
k1 = 1; k2 = 0.1; c1 = 1; c2 = -0.1; m = 1; 
pTrainGrid = 0.2; vTrainGrid = 0.2;
pTestGrid = 0.01; vTestGrid = 0.01;

% generate data
[pTrain, vTrain, aTrain] = dataGeneration(k1,k2,c1,c2,m,pTrainGrid,vTrainGrid);
[pTest, vTest, aTrue] = dataGeneration(k1,k2,c1,c2,m,pTestGrid,vTestGrid);

% vanilla interpolation
xTrain = [];
yTrain = [];
for ii = 1:length(pTrain)
    for jj = 1:length(vTrain)
        xTrain = [xTrain; pTrain(ii), vTrain(jj)];
        yTrain = [yTrain; aTrain(ii,jj)];
    end
end

sig_l = 0.1;
sig_f = 1;
KK = customKernel(xTrain,xTrain,sig_l,sig_f);

xTest = [];
yTrue = [];
for ii = 1:length(pTest)
    for jj = 1:length(vTest)
        xTest = [xTest; pTest(ii), vTest(jj)];
        yTrue = [yTrue; aTrue(ii,jj)];
    end
end

yTestIntp = zeros(length(xTest),1);
for ii = 1:length(xTest)
    yTestIntp(ii) = customKernel(xTest(ii,:),xTrain,sig_l,sig_f)/KK*yTrain;
end

yTestIntpMesh = zeros(length(pTest));

for ii = 1:length(pTest)
    for jj = 1:length(vTest)
        yTestIntpMesh(ii,jj) = customKernel([pTest(ii), vTest(jj)],xTrain,sig_l,sig_f)/KK*yTrain;
    end
end

% yTestIntpMesh = reshape(yTestIntp, length(pTest),length(pTest));

% yTestIntpMesh = zeros(length(pTest));
% for ii = 1:length(yTestIntp)
%     q = floor((ii-1)/length(pTest));
%     r = mod(ii-1,length(pTest));
%     yTestIntpMesh(r+1,q+1) = yTestIntp(ii);
% end

RMSEIntp = sqrt((yTestIntp - yTrue)'*(yTestIntp - yTrue)/length(yTestIntp));

% semiparametric interpolation
% psi1 = p, psi2 = v

Psi = xTrain;    % correct basis

if flag == 1
    Psi_2 = xTrain;
else
    Psi_2 = xTrain.^2;
end

tmp1 = Psi_2'/KK*Psi_2;
bHat = tmp1\Psi_2'/KK*yTrain;
aHat = KK\(yTrain - Psi_2*bHat);

yTestSemi = zeros(length(xTest),1);
for ii = 1:length(xTest)
    yTestSemi(ii) = customKernel(xTest(ii,:),xTrain,sig_l,sig_f)*aHat + xTest(ii,:)*bHat;
end

yTestSemiMesh = zeros(length(pTest));
for ii = 1:length(pTest)
    for jj = 1:length(vTest)
        yTestSemiMesh(ii,jj) = customKernel([pTest(ii), vTest(jj)],xTrain,sig_l,sig_f)*aHat + [pTest(ii), vTest(jj)]*bHat;
    end
end

bLin = (Psi'*Psi)\Psi'*yTrain;
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

RMSESemi = sqrt((yTestSemi - yTrue)'*(yTestSemi - yTrue)/length(yTestIntp));
RMSELin = sqrt((yTestLin - yTrue)'*(yTestLin - yTrue)/length(yTestIntp));

% % best hyperparameter selection - fitrgp
% gprmdl = fitrgp(xTrain, yTrain, 'KernelFunction', 'squaredexponential', 'BasisFunction', 'none');
% yTestGPR = predict(gprmdl,xTest); 
% RMSEGPR = sqrt((yTestGPR - yTrue)'*(yTestGPR - yTrue)/length(yTestIntp));

% mesh plot
% figure(1);
% subplot(1,3,1)
% mesh(pTest,vTest,aTrue,'EdgeColor',"green");
% hold on;
% mesh(pTest,vTest,yTestIntpMesh,'EdgeColor',"cyan");
% axis([-0.7 -0.5 -0.8 -0.6 1 1.5])
% legend('true', 'KI',fontsize=20)
% subplot(1,3,2)
% mesh(pTest,vTest,aTrue,'EdgeColor',"green");
% hold on;
% mesh(pTest,vTest,yTestSemiMesh,'EdgeColor',"red");
% axis([-0.7 -0.5 -0.8 -0.6 1 1.5])
% legend('true', 'SKI',fontsize=20)
% subplot(1,3,3)
% mesh(pTest,vTest,aTrue,'EdgeColor',"green");
% hold on;
% mesh(pTest,vTest,yTestLinMesh,'EdgeColor',"blue");
% axis([-0.7 -0.5 -0.8 -0.6 1 1.5])
% legend('true', 'LS',fontsize=20)

figure(2);
mesh(pTest,vTest,abs(yTestSemiMesh - aTrue),'EdgeColor',"red","LineWidth",1.5);
hold on;
mesh(pTest,vTest,abs(yTestIntpMesh - aTrue),'EdgeColor',"cyan","LineWidth",1.5);
mesh(pTest,vTest,abs(yTestLinMesh - aTrue), 'EdgeColor','blue',"LineWidth",1.5);
legend('SKI error', 'KI error', 'LS error',fontsize=40)
xlabel('$p$','Interpreter','latex')
ylabel('$v$','Interpreter','latex')
zlabel('absolute error')
set(gca,"FontSize",40)