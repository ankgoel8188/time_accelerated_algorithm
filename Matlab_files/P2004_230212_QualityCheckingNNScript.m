%% P20.04: Quality Checking NN Script
% Author: Turibius Rozario
% Advisor: Dr. Ankit Goel
% Date: 2023-02-12
% Generates y using a know NN (randomized theta), and later attempts to
% optimize a new NN to determine if the existing NN script by Turibius
% operates as expected or not. A simple structure containing 1 layer and 2
% neurons is first used, and then a more complicated system.

clear; clc; clf
rng(1)
%% Initials
X = rand(1, 500) * 5;
layers = [4, 2, 1];
epochs = 30;
alpha = 0.0005;
bias = 1;
activFunc = ["relu", "none"];
labels = ["batch", "stochastic"];
ntests = length(labels);
% optimizer = "batch";
%% Generating Architecture and True THETA and Y
nl = length(layers); % num of layers
ns = size(X, 2); % num of samples
activFunc = repelem(activFunc, [nl-1, 1]); % replicating the strings
THETA = cell(1, nl); % initializing cell
Xn = cell(1, nl+1); % initializing cell
Xn{1} = X;
% Initializing THETA with random values
for i = 1 : nl
    if bias == 1
        THETA{i} = [randn(size(Xn{i}, 1), layers(i));
                randn(1, layers(i))];
    end
    Xn{i+1} = NeuralLayer(Xn{i}, THETA{i}, activFunc(i), bias);
end
NN = @(x, THETA) NNfull(x, THETA, layers, activFunc, bias);
Xn = NN(X, THETA);
Y = Xn{end};
COSTS = cell(1, ntests);
THETA_hat_norms = zeros(nl, ntests);
%% Training on simple model: Batch
[~, THETA_hat, Costs] = GradientDescent(X, Y, layers, activFunc, ...
    bias, alpha, epochs, labels(1), THETA);
COSTS{1} = Costs;
for ii = 1 : length(THETA_hat)
    THETA_hat_norms(ii, 1) = norm(THETA_hat{ii});
end
%% Training on simple model: Stochastic
[~, THETA_hat, Costs] = GradientDescent(X, Y, layers, activFunc, ...
    bias, alpha, epochs, labels(2), THETA);
for ii = 1 : length(THETA_hat)
    THETA_hat_norms(ii, 2) = norm(THETA_hat{ii});
end
extractor = 1 : ns : length(Costs);
COSTS{2} = Costs(extractor);
%% Plot results
figure(1)
semilogy(COSTS{1})
hold on
semilogy(COSTS{2})
grid on;
ax = gca;
ax.TickLabelInterpreter='latex';
aa = legend(labels);
set(aa, 'box','off','location','southwest','interpreter','latex');
xlabel("Iterations");
ylabel("$J$");
% For LaTeX files, comment the two lines:
% title("Convergence Rates", Interpreter="latex");
% title(subTitle, Interpreter="latex");
hold off
% And then do:
% export_fig FiguresLaTeX\P2004_ConvergenceRates -transparent -eps