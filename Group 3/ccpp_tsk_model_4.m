% Fuzzy Systems 2018 - Group 3 - Ser06
% Kosmas Tsiakas 8255
% Regression with TSK models 
% Cobined Cycle Power Plant (CCPP) dataset from UCI repository
tic
%% CLEAR
clear all;
close all;

%% BEGIN
fprintf('\n *** begin %s ***\n\n',mfilename);

%% READ DATA
load CCPP.dat
% COLUMNS -> 1:T, 2:AP, 3:RH, 4:V, 5:energy_output 

%% SPLIT DATASET
fprintf('\n *** Dataset splitting\n');

training_data = CCPP(1 : round(0.6*9568), :); % 60% of the dataset is for training
validation_data = CCPP(round(0.6*9568)+1 : round(0.8 * 9568), :); % 20% is for evaluation
check_data = CCPP(round(0.8*9568)+1 : end, :); % 20% is for testing

% TRAIN TSK MODEL

%% MODEL 2  - 3 MF - SINGLETON OUTPUT
fprintf('\n *** TSK Model 4\n');

% Set the options, 
opt = genfisOptions('GridPartition');
opt.NumMembershipFunctions = [3 3 3 3]; % Two mf for each input variable
opt.InputMembershipFunctionType = ["gbellmf" "gbellmf" "gbellmf" "gbellmf"]; % Bell-shaped
opt.OutputMembershipFunctionType = 'linear';

% Generate the FIS
fprintf('\n *** Generating the FIS\n');
init_fis = genfis(training_data(:, 1:4), training_data(:, 5), opt);

% Plot the input membership functions
figure;
[x,mf] = plotmf(init_fis,'input',1);
subplot(2,2,1);

plot(x,mf);
xlabel('input 1 - T (gbellmf)');

[x,mf] = plotmf(init_fis,'input',2);
subplot(2,2,2);
plot(x,mf);
xlabel('input 2 - AP (gbellmf)');

[x,mf] = plotmf(init_fis,'input',3);
subplot(2,2,3);
plot(x,mf);
xlabel('input 3 - RH (gbellmf)');

[x,mf] = plotmf(init_fis,'input',4);
subplot(2,2,4);
plot(x,mf);
xlabel('input 4 - V (gbellmf)');

suptitle('TSK model 4 : membership functions before training');
saveas(gcf, 'TSK_model_4/mf_before_training.png');

% Tune the fis
fprintf('\n *** Tuning the FIS\n');
% Set some options
% The fis structure already exists
% set the validation data to avoid overfitting
% display training progress information
anfis_opt = anfisOptions('InitialFIS', init_fis, 'EpochNumber', 600, 'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, 'ValidationData', validation_data);

[trn_fis, trainError, stepSize, chkFIS, chkError] = anfis(training_data, anfis_opt);

% Evaluate the fis
fprintf('\n *** Evaluating the FIS\n');
% No need to specify specific options for this, keep the defaults
output = evalfis(check_data(:,1:4),chkFIS);

%% METRICS
error = output - check_data(:,5);

mse = (1/length(error)) * sum(error.^2);

rmse = sqrt(mse);

SSres = sum( (check_data(:,5) - output).^2 );
SStot = sum( (check_data(:,5) - mean(check_data(:,5))).^2 );
r2 = 1 - SSres / SStot;

nmse = var(error) / var(check_data(:,5));

ndei = sqrt(nmse);

% Plot the metrics
figure;
plot(1:length(check_data),check_data(:,5),'*r',1:length(check_data),output, '.b');
title('Output');
legend('Reference Outputs','Model Outputs');
saveas(gcf,'TSK_model_4/output.png')

figure;
plot(error);
title('Prediction Errors');
saveas(gcf,'TSK_model_4/error.png')

figure;
plot(1:length(trainError),trainError,1:length(trainError),chkError);
title('Learning Curve');
legend('Traning Set', 'Check Set');
saveas(gcf,'TSK_model_4/learningcurves.png')

% Plot the input membership functions after training
figure;
[x,mf] = plotmf(chkFIS,'input',1);
subplot(2,2,1);

plot(x,mf);
xlabel('input 1 - T');

[x,mf] = plotmf(chkFIS,'input',2);
subplot(2,2,2);
plot(x,mf);
xlabel('input 2 - AP');

[x,mf] = plotmf(chkFIS,'input',3);
subplot(2,2,3);
plot(x,mf);
xlabel('input 3 - RH');

[x,mf] = plotmf(chkFIS,'input',4);
subplot(2,2,4);
plot(x,mf);
xlabel('input 4 - V');

suptitle('TSK model 4 : membership functions after training');
saveas(gcf, 'TSK_model_4/mf_after_training.png');


fprintf('MSE = %f RMSE = %f R^2 = %f NMSE = %f NDEI = %f\n', mse, rmse, r2, nmse, ndei)

toc

%% MSE = 16.873637 RMSE = 4.107753 R^2 = 0.941565 NMSE = 0.058422 NDEI = 0.241707
%% Elapsed time is 4837.748601 seconds.