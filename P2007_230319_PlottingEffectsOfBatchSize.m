%% P20.07: PlottingEffectsOfBatchSize
% Author: Turibius Rozario
% Advisor: Dr. Ankit Goel
% Date: 2023-03-19
% Plots results from P2007 ... .py, whose outputs are in the
% Python/P2007_Files/ directory.

clear; clc; close all

% Parameters
FILE_LOC = "Python\P2007_Files\TEST_";
TESTS = 4;
POW = 6; % Must be same as in P2007...py file
RUNS = 20; % Must be same as in P2007...py file
epochs = 1:100;
test_types = ["$\alpha=0.01$", "$\alpha=0.01$, Adam"
            "$\alpha=0.01 \times n_b$", "$\alpha = 0.01 \times n_b$, Adam"];
ylimit = [1e-8, 1e2];

% Initializations
NORMS = cell(TESTS);
COSTS_ALL= cell(TESTS, POW + 1);
FINAL_COSTS = cell(TESTS, POW + 1);
% Which runs are the best or worst
best_idx = zeros(TESTS, POW + 1);
worst_idx = best_idx;
% These will contain the full data regarding the cost
BEST_COSTS = FINAL_COSTS;
WORST_COSTS = FINAL_COSTS;
AVG_COSTS = FINAL_COSTS;
labels_bestworst = string(POW + 1);
for ii=0:POW
    labels_bestworst(ii+1) = "$n_b = " + string(2 ^ ii) + "$";
end
%% With Outliers
% Filling NORMS and COSTS, using actual data
for ii=1:TESTS
    file = FILE_LOC + string(ii) + "_norms.csv";
    NORMS{ii} = readmatrix(file);
    file = FILE_LOC + string(ii) + "_J_";
    for ij=1:POW + 1
        file_J = file + string(ij - 1) + ".csv";
        COSTS_ALL{ii, ij} = readmatrix(file_J);
    end
end

% Finding the final costs of each run, in each test and batch size
for ii=1:TESTS
    for ij=1:POW+1
        FINAL_COSTS{ii, ij} = COSTS_ALL{ii, ij}(:, end);
        [~, best_idx(ii, ij)] = min(FINAL_COSTS{ii, ij});
        [~, worst_idx(ii, ij)] = max(FINAL_COSTS{ii, ij});
        BEST_COSTS{ii, ij} = COSTS_ALL{ii, ij}(best_idx(ii, ij), :);
        WORST_COSTS{ii, ij} = COSTS_ALL{ii, ij}(worst_idx(ii, ij), :);
    end
end

figure("Name", "Evolution of Best Runs")
plotEvolutions(epochs, BEST_COSTS, POW, labels_bestworst, test_types, ...
    ylimit);
figure("Name", "Evolution of Worst Runs")
plotEvolutions(epochs, WORST_COSTS, POW, labels_bestworst, test_types, ...
    ylimit);
f = figure("Name", "Final Costs for All Runs");
plotFinalCosts(FINAL_COSTS, test_types);
exportgraphics(f, "FiguresLaTeX\P2007_AllRunsScatter.png")

%% Average Plot
for ii=1:TESTS
    for ij=1:POW+1
        AVG_COSTS{ii, ij} = mean(COSTS_ALL{ii, ij}, 1);
    end
end
f = figure("Name", "Average of All Runs");
plotEvolutions(epochs, AVG_COSTS, POW, labels_bestworst, test_types, ...
    ylimit);
exportgraphics(f, "FiguresLaTeX\P2007_AvgRuns.png")

%% Outlier Removal
for ii=1:TESTS
    for ij=1:POW+1
        [~, idx] = rmoutliers(FINAL_COSTS{ii, ij});
        FINAL_COSTS{ii, ij}(idx) = NaN;
        [~, best_idx(ii, ij)] = min(FINAL_COSTS{ii, ij});
        [~, worst_idx(ii, ij)] = max(FINAL_COSTS{ii, ij});
        BEST_COSTS{ii, ij} = COSTS_ALL{ii, ij}(best_idx(ii, ij), :);
        WORST_COSTS{ii, ij} = COSTS_ALL{ii, ij}(worst_idx(ii, ij), :);
    end
end

figure("Name", "Evolution of Best Runs, Outliers Removed")
plotEvolutions(epochs, BEST_COSTS, POW, labels_bestworst, test_types, ...
    ylimit);
figure("Name", "Evolution of Worst Runs, Outliers Removed")
plotEvolutions(epochs, WORST_COSTS, POW, labels_bestworst, test_types, ...
    ylimit);
f = figure("Name", "Final Costs for All Runs, Outliers Removed");
plotFinalCosts(FINAL_COSTS, test_types);
exportgraphics(f, "FiguresLaTeX\P2007_AllRunsScatterNO.png")
%% Functions
function plotTheRest(epoch_vector, plot_cell, test_type, iterator_max)
    semilogy(epoch_vector, plot_cell{test_type, 1});
    hold on
    for ii = 2:iterator_max
        semilogy(epoch_vector, plot_cell{test_type, ii})
    end
end
function plotBestOrWorst(epoch_vector, plot_cell, test_type, ...
                            iterator_max, labels, figtitle, ylimit)
    plotTheRest(epoch_vector, plot_cell, test_type, iterator_max)
    grid on
    ax = gca;
    ax.TickLabelInterpreter='latex';
    aa = legend(labels);
    set(aa, 'box','off','location','northeast','interpreter','latex');
    xlabel("Epochs");
    ylabel("$J$");
    title(figtitle, Interpreter="latex")
    ylim(ylimit)
    hold off
end
function scatterEndResults(cost_cell, figtitle)
    iterator_max = length(cost_cell);
    power_vector = zeros(1, iterator_max);
    cost_matrix = zeros(length(cost_cell{1}), iterator_max);
    for ii = 1 : iterator_max
        power_vector(ii) = 2 ^ (ii - 1);
        cost_matrix(:, ii) = cost_cell{ii};
    end
    scatter(power_vector, cost_matrix, "*")
    grid on
    ax = gca;
    ax.TickLabelInterpreter='latex';
    xlabel("$n_b$");
    ylabel("$J$");
    xticks(power_vector);
    set(gca, 'Yscale', 'log')
    set(gca, 'Xscale', 'log')
    title(figtitle, Interpreter="latex")
end
function plotEvolutions(epochs, COSTS, POW, labels_bestworst, ...
test_types, ylimit)
    subplot(2, 2, 1)
    plotBestOrWorst(epochs, COSTS, 1, POW + 1, labels_bestworst, ...
        test_types(1), ylimit)
    subplot(2, 2, 2)
    plotBestOrWorst(epochs, COSTS, 3, POW + 1, labels_bestworst, ...
        test_types(3), ylimit)
    subplot(2, 2, 3)
    plotBestOrWorst(epochs, COSTS, 2, POW + 1, labels_bestworst, ...
        test_types(2), ylimit)
    subplot(2, 2, 4)
    plotBestOrWorst(epochs, COSTS, 4, POW + 1, labels_bestworst, ...
        test_types(4), ylimit)
    set(gcf, 'Position', get(0, 'Screensize'));
end
function plotFinalCosts(FINAL_COSTS, test_types)
    test = 1;
    subplot(2, 2, test);
    scatterEndResults(FINAL_COSTS(test, :), test_types(test));
    test = 2;
    subplot(2, 2, test + 1);
    scatterEndResults(FINAL_COSTS(test, :), test_types(test));
    test = 3;
    subplot(2, 2, test - 1);
    scatterEndResults(FINAL_COSTS(test, :), test_types(test));
    test = 4;
    subplot(2, 2, test);
    scatterEndResults(FINAL_COSTS(test, :), test_types(test));
    set(gcf, 'Position', get(0, 'Screensize'));
end
