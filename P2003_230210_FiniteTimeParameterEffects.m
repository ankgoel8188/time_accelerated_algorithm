%% P20.03: Effects of Parameters in Finite Time Estimator
% Author:   Turibius Rozario
% Advisor:  Dr. Ankit Goel
% Tests the effects of lambda1, lambda2, alpha1, and alpha2 on the
% finite-time optimizer proposed by Dr. Garg. The simulink model used for 
% this is P2003_230208_FiniteTimeEstimator1.slx

clear; clc;

thetaStar = 2;
% Base parameters:
Lambda1 = 1.5;
Lambda2 = 1.5;
Alpha1 = 0.5;
Alpha2 = 2.5;
% Range of changes:
deltaLambda = 0.5;
deltaAlpha1 = 0.2;
deltaAlpha2 = 0.5;

% Initialization
t = cell(1, 5);
errors = t;
labels = strings(1, 5);
% CHANGE THIS for TEXT:
% labelStart = "$\lambda_1 = ";
labelStart = "$\lambda_2 = ";
% labelStart = "$\alpha_1 = ";
% labelStart = "$\alpha_2 = ";
subTitle = "Base: $\lambda_1 = " + string(Lambda1) + ", \lambda_2 = " + ...
    string(Lambda2) + ", \alpha_1 = " + string(Alpha1) + ...
    ", \alpha_2 = " + string(Alpha2) + "$";

lambda1 = Lambda1;
lambda2 = Lambda2;
alpha1 = Alpha1;
alpha2 = Alpha2;

for ii = 1 : 5
    % CONTROL PARAMETER:
%     lambda1 = Lambda1 + deltaLambda * (ii - 1) - deltaLambda * 2;
%     labels(ii) = labelStart + string(lambda1) + "$";
    lambda2 = Lambda2 + deltaLambda * (ii - 1) - deltaLambda * 2;
    labels(ii) = labelStart + string(lambda2) + "$";
%     alpha1 = Alpha1 + deltaAlpha1 * (ii - 1) - deltaAlpha1 * 2;
%     labels(ii) = labelStart + string(alpha1) + "$";
%     alpha2 = Alpha2 + deltaAlpha2 * (ii - 1) - deltaAlpha2 * 2;
%     labels(ii) = labelStart + string(alpha2) + "$";

    theta = sim("P2003_230208_FiniteTimeEstimator1.slx");
    t{ii} = theta.tout;
    errors{ii} = theta.Finite;
    errors{ii} = thetaTildeAbs(errors{ii}, thetaStar);   
end

LoadFigurePrintingProperties
figure(1)
semilogy(t{1}, errors{1});
ylim([1e-10, 2]);
hold on
for ii = 2 : 5
    semilogy(t{ii}, errors{ii});
end
grid on;
ax = gca;
ax.TickLabelInterpreter='latex';
aa = legend(labels);
set(aa, 'box','off','location','northeast','interpreter','latex');
xlabel("Time (s)");
ylabel("$|\tilde{\theta}_{\mathrm{FT}}|$");
% For LaTeX files, comment the two lines:
% title("Effects of Parameters on Finite-Time Method", Interpreter="latex");
% subtitle(subTitle, Interpreter="latex");
hold off
% And then do:
export_fig FiguresLaTeX\P2003_EffectsOfParametersLambda2 -transparent -pdf

function error = thetaTildeAbs(thetaHat, thetaStar)
    error = abs(thetaHat - thetaStar);
end
