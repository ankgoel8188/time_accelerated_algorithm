%% P20.05: Effects of P0 on EM Estimator
% Author:   Turibius Rozario
% Advisor:  Dr. Ankit Goel
% Date:     2023-02-15
% Tests the effects of P0 on the exact minimizer solution to the cost
% function with forgetting.

clear; clc;

% Initialization
t = cell(1, 6);
errors = t;
labels = strings(1, 6);
subTitle = "Base: $\lambda = 1,\, A_0 = [0; 0],\, P_0 = p_0 \times I_2$";

for ii = 1 : 6
    p0 = 0.2 * (ii - 1);
    labels(ii) = "$p_0 = " + string(p0) + "$";

    theta = sim("P2005_230215_CumulativeCostEMEffectsVector.slx");
    t{ii} = theta.tout;
    errors{ii} = theta.thetaTilde;
end

LoadFigurePrintingProperties
figure(1)
semilogy(t{1}, errors{1});
hold on
for ii = 2 : length(t)
    semilogy(t{ii}, errors{ii});
end
grid on;
ax = gca;
ax.TickLabelInterpreter='latex';
aa = legend(labels);
set(aa, 'box','off','location','southwest','interpreter','latex');
xlabel("Time (s)");
ylabel("$||\tilde{\theta}_{\mathrm{EM}}||_2$");
% For LaTeX files, comment the two lines:
% title("Effects of Parameters on Finite-Time Method", Interpreter="latex");
title(subTitle, Interpreter="latex");
hold off
% And then do:
export_fig FiguresLaTeX\P2005_EffectsOfp0OnEM -transparent -eps

