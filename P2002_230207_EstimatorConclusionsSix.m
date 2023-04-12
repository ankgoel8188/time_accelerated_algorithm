%% P20.02: Estimator Conclusions Six
% Author:   Turibius Rozario
% Advisor:  Dr. Ankit Goel
% Date:     2023-02-07
% Runs the P20.02 Master Testing simulation of the 3 different cost
% computations and 2 different minimization techniques (6 results). A
% figure is later plotted, with an exponential reference. Note that
% absolute value on the cost is needed because the 'cost' is actually 
% thetaTilde. The conclusions are:
% GD with instantaneous cost
% GD with cumulative cost
% GD with cumulative cost (forgetting, lambda > 0)
% EM with instantaneous cost
% EM with cumulative cost
% EM with cumulative cost (forgetting, lambda > 0)

clear; clc;
thetaStar = 2;
theta = sim("P2002_230207_MasterTest01.slx");
labels = ["$\tilde{\theta}_\mathrm{EM}$ with $J_i$", ...
    "$\tilde{\theta}_\mathrm{EM}$ with $J_c$, $\lambda = 0$", ...
    "$\tilde{\theta}_\mathrm{EM}$ with $J_c$, $\lambda = 1$", ...
    "$\tilde{\theta}_\mathrm{GD}$ with $J_i$", ...
    "$\tilde{\theta}_\mathrm{GD}$ with $J_c$, $\lambda = 0$", ...
    "$\tilde{\theta}_\mathrm{GD}$ with $J_c$, $\lambda = 1$"
    ];

t = theta.tout;
errors = [theta.InstEM, theta.CumulEM, theta.CumulEMlambda, ...
    theta.InstGD, theta.CumulGD, theta.CumulGDlambda];
errors = thetaTildeAbs(errors, thetaStar);

LoadFigurePrintingProperties
figure(Name="Main")
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
