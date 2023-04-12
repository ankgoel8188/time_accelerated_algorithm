%% P20.03: Estimator Conclusions With Finite
% Author:   Turibius Rozario
% Advisor:  Dr. Ankit Goel
% Date:     2023-02-08
% Runs the P20.03 Master Testing simulation of the 3 different cost
% computations and 2 different minimization techniques (5 results, since
% instantaneous cost exact minimizer is not used). The finite time method
% is used in this case. A figure is later plotted, with an exponential 
% reference. Note that absolute value on the cost is needed because the 
% 'cost' is actually thetaTilde. The conclusions are:
% GD with instantaneous cost
% GD with cumulative cost
% GD with cumulative cost (forgetting, lambda > 0)
% EM with cumulative cost
% EM with cumulative cost (forgetting, lambda > 0)
% Finite Time

clear; clc;
thetaStar = 2;
theta = sim("P2003_230208_MasterTest01.slx");
labels = ["$\tilde{\theta}_\mathrm{FT}$", ...
%     "$\tilde{\theta}_\mathrm{EM}$ with $J_c$, $\lambda = 0$", ...
%     "$\tilde{\theta}_\mathrm{EM}$ with $J_c$, $\lambda = 1$", ...
%     "$\tilde{\theta}_\mathrm{GD}$ with $J_i$", ...
    "$\tilde{\theta}_\mathrm{GD}$ with $J_c$, $\lambda = 0$", ...
    "$\tilde{\theta}_\mathrm{GD}$ with $J_c$, $\lambda = 1$"
    ];

t = theta.tout;
errors = [theta.Finite, ...
            theta.CumulEM, theta.CumulEMlambda, ...
            theta.InstGD, theta.CumulGD, theta.CumulGDlambda];
errors = thetaTildeAbs(errors, thetaStar);

LoadFigurePrintingProperties
figure(1)
semilogy(t, errors(:, 1));
hold on
for ii = 2:size(errors, 2)
    semilogy(t, errors(:, ii));
end
grid on;
ax = gca;
ax.TickLabelInterpreter='latex';
aa = legend(labels);
set(aa, 'box','off','location','southwest','interpreter','latex');
xlabel("Time (s)");
ylabel("$|\tilde{\theta}|$");
hold off

function error = thetaTildeAbs(thetaHat, thetaStar)
error = abs(thetaHat - thetaStar);
end
