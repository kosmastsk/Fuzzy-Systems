 % Fuzzy Systems 2018 - Group 3 - Ser06
% Kosmas Tsiakas 8255
% Regression with TSK models - Substractive Clustering
% Bank dataset from delve repository
tic
%% CLEAR
clear all;
close all;

%% BEGIN
fprintf('\n *** begin %s ***\n\n', mfilename);

%% READ DATA
load Bank.data

%% SHUFFLE DATA
shuffledBank = zeros(size(Bank));
rand_pos = randperm(length(Bank)); %array of random positions
% new array with original data randomly distributed
for k = 1:length(Bank)
    shuffledBank(k, :) = Bank(rand_pos(k), :);
end

% Keep only the number of features we want and not all of them
% Specify their order and later use the ranks array
[ranks, weights] = relieff(Bank(:, 1:32), Bank(:, end), 100);

%% FINAL TSK MODEL
fprintf('\n *** TSK Model with 15 features and radii 1 - Substractive Clustering\n');

f = 15;
radii = 1;

%% SPLIT DATASET
fprintf('\n *** Dataset splitting\n');

training_data = shuffledBank(1 : round(0.6 * 8192), :); % 60% of the dataset is for training
validation_data = shuffledBank(round(0.6 * 8192) + 1 : round(0.8 * 8192), :); % 20% is for evaluation
check_data = shuffledBank(round(0.8 * 8192) + 1 : end, :); % 20% is for testing

%% NORMALIZE DATA
% Normalize each set differently so that they are separated through the
% whole process
for i = 1 : size(training_data,2)-1
    training_data_min = min(training_data(:,i));
    training_data_max = max(training_data(:,i));
    training_data(:,i) = (training_data(:,i) - training_data_min) / (training_data_max - training_data_min); % Scaled to [0, 1]
    training_data(:,i) = training_data(:,i) * 2 - 1;

    validation_data(:,i) = (validation_data(:,i) - training_data_min) / (training_data_max - training_data_min); % Scaled to [0, 1]
    validation_data(:,i) = validation_data(:,i) * 2 - 1;

    check_data(:,i) = (check_data(:,i) - training_data_min) / (training_data_max - training_data_min); % Scaled to [0, 1]
    check_data(:,i) = check_data(:,i) * 2 - 1;
end

training_data_x = training_data(:,ranks(1:f));
training_data_y = training_data(:,end);

validation_data_x = validation_data(:,ranks(1:f));
validation_data_y = validation_data(:,end);

check_data_x = check_data(:,ranks(1:f));
check_data_y = check_data(:,end);

%% TRAIN TSK MODEL

%% MODEL WITH 21 FEATURES AND 4 RULES

% Generate the FIS
fprintf('\n *** Generating the FIS\n');

% As input data I give the train_id's that came up with the
% partitioning and only the most important features
% As output data is just the last column of the test_data that
% are left
init_fis = genfis2(training_data_x, training_data_y, radii);
rules = length(init_fis.rule)
% Plot some input membership functions
figure;
for i = 1 : f
    [x, mf] = plotmf(init_fis, 'input', i);
    plot(x,mf);
    hold on;
end
title('Membership functions before training');
xlabel('x');
ylabel('Degree of membership');
saveas(gcf, 'Final_TSK_model/mf_before_training.png');

figure;
[x, mf] = plotmf(init_fis, 'input', 1);
subplot(2,2,1);
plot(x,mf);
xlabel('input 1');

[x, mf] = plotmf(init_fis, 'input', 2);
subplot(2,2,2);
plot(x,mf);
xlabel('input 2');

[x, mf] = plotmf(init_fis, 'input', 3);
subplot(2,2,3);
plot(x,mf);
xlabel('input 3');

[x, mf] = plotmf(init_fis, 'input', 4);
subplot(2,2,4);
plot(x,mf);
xlabel('input 4');

suptitle('Final TSK model : some membership functions before training');
saveas(gcf, 'Final_TSK_model/some_mf_before_training.png');

% Tune the fis
fprintf('\n *** Tuning the FIS\n');

% Set some options
% The fis structure already exists
% set the validation data to avoid overfitting
anfis_opt = anfisOptions('InitialFIS', init_fis, 'EpochNumber', 150, 'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, 'DisplayStepSize', 0, 'DisplayFinalResults', 0, 'ValidationData', [validation_data_x validation_data_y]);

[trn_fis, trainError, stepSize, chkFIS, chkError] = anfis([training_data_x training_data_y], anfis_opt);

% Evaluate the fis
fprintf('\n *** Evaluating the FIS\n');

% No need to specify specific options for this, keep the defaults
output = evalfis(check_data_x, chkFIS);

%% METRICS
error = output - check_data_y;

mse = (1 / length(error)) * sum(error .^ 2);

rmse = sqrt(mse);

SSres = sum((check_data_y - output) .^ 2);
SStot = sum((check_data_y - mean(check_data_y)) .^ 2);
r2 = 1 - SSres / SStot;

nmse = var(error) / var(check_data_y);

ndei = sqrt(nmse);

% Plot the metrics
figure;
plot(1:length(check_data_x), check_data_y, '*r', 1:length(check_data_y), output, '.b');
title('Output');
legend('Reference Outputs', 'Model Outputs');
saveas(gcf, 'Final_TSK_model/output.png')

figure;
plot(error);
title('Prediction Errors');
saveas(gcf, 'Final_TSK_model/error.png')

figure;
plot(1:length(trainError), trainError, 1:length(chkError), chkError);
title('Learning Curve');
legend('Traning Set', 'Check Set');
saveas(gcf, 'Final_TSK_model/learningcurves.png')

% Plot the input membership functions after training
figure;
for i = 1 : f
    [x, mf] = plotmf(chkFIS, 'input', i);
    plot(x,mf);
    hold on;
end
title('Membership functions after training');
xlabel('x');
ylabel('Degree of membership');
saveas(gcf, 'Final_TSK_model/mf_after_training.png');

figure;
[x, mf] = plotmf(chkFIS, 'input', 1);
subplot(2,2,1);
plot(x,mf);
xlabel('input 1');

[x, mf] = plotmf(chkFIS, 'input', 2);
subplot(2,2,2);
plot(x,mf);
xlabel('input 2');

[x, mf] = plotmf(chkFIS, 'input', 3);
subplot(2,2,3);
plot(x,mf);
xlabel('input 3');

[x, mf] = plotmf(chkFIS, 'input', 4);
subplot(2,2,4);
plot(x,mf);
xlabel('input 4');

suptitle('Final TSK model : some membership functions after training');
saveas(gcf, 'Final_TSK_model/after.png');

fprintf('MSE = %f RMSE = %f R^2 = %f NMSE = %f NDEI = %f\n', mse, rmse, r2, nmse, ndei)

toc
%% MSE = 0.004832 RMSE = 0.069511 R^2 = 0.837169 NMSE = 0.162823 NDEI = 0.403513
%% Elapsed time is 102.861282 seconds.