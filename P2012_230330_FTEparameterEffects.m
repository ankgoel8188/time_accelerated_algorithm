%% P20.12: Effect of various parameters on FTE
% Author:   Turibius Rozario
% Advisor:  Dr. Ankit Goel
% Date:     2023-03-30
% Ensure to call P2012...py first. Data files should be located in
% Pytorch\P2012\. Parameters that are printed were parameters initially
% used in P2012...py.

% clear; clc; close all

% Parameters
LAYER_TYPE = 3; % 1=1 to 1; 2=10 to 1, 3=10 to 5; #=neuron in to neuron out
SELECT_TEST = 5; % 1=alpha_1, alpha_2, c_1, c_2, 5=DelT
FILE_LOC = "Pytorch\P2012\";
FIGURE_LOC = "FiguresLaTeX\P2012_ParameterEffectsOnFTE\";
label_NI = ["1", "10", "10"];
label_NO = ["1", "1", "5"];
test_parameters = ["$\alpha_1$", "$\alpha_2$", "$c_1$", "$c_2$", ...
    "$\Delta T$"];
test_parameters_values = {
    [0.5, 0.6125, 0.725, 0.8375, 0.95] % alpha_1
    [2.5, 2.1375, 1.775, 1.4125, 1.05] % alpha_2
    fliplr([0.5, 1.0, 1.5, 2.0, 2.5]) % c_1
    fliplr([0.5, 1.0, 1.5, 2.0, 2.5]) % c_2
    [0.01, 0.04, 0.16, 0.64, 2.56] % DelT
};
test_types = ["alpha_1", "alpha_2", "c_1", "c_2", "DelT"];
lower_lim = [1e-14, 1e-14, 1e-14];
lower_lim = lower_lim(LAYER_TYPE);
test_types = test_types(SELECT_TEST);
test_parameters_values = test_parameters_values{SELECT_TEST};
epochs = 1:1000;

labels = string(zeros(1, length(test_parameters)));
for ii = 1 : length(labels)
    labels(ii) = test_parameters(SELECT_TEST) + "=" + ...
        test_parameters_values(ii);
end

all_costs = cell(5, 1);
for ii = 1:5
    curr_filename = FILE_LOC + string(LAYER_TYPE) + "_" + ...
        test_types + "_" + string(ii-1) + ".csv";
    all_costs{ii} = readmatrix(curr_filename);
end

LoadFigurePrintingProperties
f = figure(1);
% f.Position(3:4) = [600 400];
semilogy(epochs * test_parameters_values(1), all_costs{1}, ...
    'LineStyle',":"); % FIX
hold on
for ii = 2 : length(test_parameters_values)
    semilogy(epochs * test_parameters_values(ii), all_costs{ii}, ...
        'LineStyle',":") % FIX
end
grid on;
ax = gca;
ax.TickLabelInterpreter='latex';
aa = legend(labels);
set(aa, 'box','off','location','southwest','interpreter','latex');
xlabel("Time (s)");
ylabel("$J$");
ylim([lower_lim, 1])
hold off
% exportgraphics(f, FIGURE_LOC + string(LAYER_TYPE) + "_" + ...
%     test_types + ".png")
% exportgraphics(f, FIGURE_LOC + string(LAYER_TYPE) + "_" + ...
%     test_types + ".eps")
