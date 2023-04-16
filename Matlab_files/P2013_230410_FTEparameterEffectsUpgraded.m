%% P20.13: Effect of various parameters on FTE (Upgraded
% Author:   Turibius Rozario
% Advisor:  Dr. Ankit Goel
% Date:     2023-04-10
% Ensure to call P2013...py first. Data files should be located in
% Pytorch\P2013\. Parameters that are printed were parameters initially
% used in P2013...py. This is like P2012, but different parameters, number
% of samples.

clear; clc; close all

% Parameters
LAYER_TYPE = 1; % 1=1 to 1; 2=10 to 1, 3=10 to 5; #=neuron in to neuron out
SELECT_TEST = 1; % 1=alpha_1, alpha_2, c_1, c_2, 5=DelT
FILE_LOC = "Pytorch\P2013\";
FIGURE_LOC = "FiguresLaTeX\P2013_ParameterEffectsOnFTE\";
label_NI = ["1", "10", "10"];
label_NO = ["1", "1", "5"];
test_parameters = ["$\alpha_1$", "$\alpha_2$", "$c_1$", "$c_2$", ...
    "$\Delta T$"];
test_parameters_values = {
    [0.01, 0.03, 0.09, 0.27, 0.81] % alpha_1
    [5, 10, 20, 40, 80] % alpha_2
    [1, 1.5, 2.0, 2.5, 3.0] % c_1
    [1, 1.5, 2.0, 2.5, 3.0] % c_2
    [1e-2, 1e-3, 1e-4, 1e-5, 1e-6] % DelT
};
lower_lim = [1e-14, 1e-14, 1e-14];

%%
% if any other test other than 5, only the first DelT should be used.
if SELECT_TEST ~= 5
    DelT = test_parameters_values{end}(1);
end

% trims the selections
test_types = ["alpha_1", "alpha_2", "c_1", "c_2", "DelT"];
lower_lim = lower_lim(LAYER_TYPE);
test_types = test_types(SELECT_TEST);
test_parameters_values = test_parameters_values{SELECT_TEST};

% generates labels
labels = string(zeros(1, length(test_parameters)));
for ii = 1 : length(labels)
    labels(ii) = test_parameters(SELECT_TEST) + "=" + ...
        test_parameters_values(ii);
end

% import the data
all_costs = cell(5, 1);
for ii = 1:5
    curr_filename = FILE_LOC + string(LAYER_TYPE - 1) + "_" + ...
        test_types + "_" + string(ii-1) + ".csv";
    all_costs{ii} = readmatrix(curr_filename);
end

%%
LoadFigurePrintingProperties
f = figure(1);
% f.Position(3:4) = [600 400];
if SELECT_TEST ~= 5
%     seconds_to_epochs = 1 / DelT;
    time_taken = length(all_costs{1}) * DelT;
%     upper_time = length(all_costs{1}) * epochs_second / DelT;
    semilogy(0:DelT:time_taken, ...
        all_costs{1}, 'LineStyle',":"); % FIX
    hold on
    for ii = 2 : length(test_parameters_values)
        semilogy(0:DelT:(length(all_costs{ii})/(epochs_second / DelT)), ...
            'LineStyle',":") % FIX
    end
else
    semilogy(1:epochs / test_parameters_values(1), all_costs{1}, ...
        'LineStyle',":"); % FIX
    hold on
    for ii = 2 : length(test_parameters_values)
        semilogy(1:epochs / test_parameters_values(ii), all_costs{ii}, ...
            'LineStyle',":") % FIX
    end
end
grid on;
ax = gca;
ax.TickLabelInterpreter='latex';
aa = legend(labels);
set(aa, 'box','off','location','southwest','interpreter','latex');
xlabel("Epochs");
ylabel("$J$");
ylim([lower_lim, 1])
hold off
% exportgraphics(f, FIGURE_LOC + string(LAYER_TYPE) + "_" + ...
%     test_types + ".png")
% exportgraphics(f, FIGURE_LOC + string(LAYER_TYPE) + "_" + ...
%     test_types + ".eps")
