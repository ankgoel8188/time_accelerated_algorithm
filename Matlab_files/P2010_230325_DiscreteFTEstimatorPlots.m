%% P20.10: Discrete Finite Time Estimator (Simple) Plots
% Author:   Turibius Rozario
% Advisor:  Dr. Ankit Goel
% Date:     2023-03-25
% Ensure to call P2010...py first! This script simply plots the results
% from the python script to compare SGD, Adam, and FTE optimizers

clear; clc; close all

% Parameters
FILE_LOC = "Pytorch\P2010\";
label_NI = ["1", "10", "10"];
label_NO = ["1", "1", "5"];
test_types = ["SGD", "Adam", "FTE"];
parameters = {"$\alpha = 0.1$",...
    "$\alpha = 0.1$", ...
    "$\alpha_1 = 0.5$, $\alpha_2 = 2.5$, " + ...
    "$c_1 = c_2 = 1.5$, $\Delta T = 0.1$"};

epochs = 1:500;
ylimit = [1e-25, 5e-1];

% Reading the files
all_costs = cell(3, 1);
for ii=1:3
    all_costs{ii} = cell(3, 1);
    curr_filename = FILE_LOC + "costs_" + test_types(ii) + "_";
    for ij=1:3
        all_costs{ii}{ij} = readmatrix(curr_filename + string(ij) +".csv");
    end
end

f = figure(1);
for ii=1:3
    subplot(2, 2, ii)
    semilogy(epochs, all_costs{1}{ii})
    hold on
    for ij=2:3
        semilogy(epochs, all_costs{ij}{ii})
    end
    grid on
    ax = gca;
    ax.TickLabelInterpreter = 'latex';
    aa = legend(test_types);
    set(aa, 'box','off','location','southwest','interpreter','latex');
    xlabel("Epochs");
    ylabel("$J$");
    title("$n_i=$" + label_NI(ii) + ", $n_o=$" + label_NO(ii));
    ylim(ylimit)
    hold off
end
subplot(2, 2, 4)
axis off
c = 0.1;
for ii=1:3
text(0, (ii - 1) * c, test_types(ii) + " parameter(s): " + ...
    parameters(ii), 'Interpreter', 'latex')
end
text(0, 3 * c, "$\hat{y} = \sigma (\mathcal{L}(x, \theta))$ " + ...
    "where $x\in \mathbf{R}^{n_i},\, y\in \mathbf{R}^{n_o}$, " + ...
    "for $n_s=80$", ...
    "Interpreter", "latex")
set(gcf, 'Position', get(0, 'Screensize'));
exportgraphics(f, "FiguresLaTeX\P2010_OptimizerComparisonWithFTE.png")
