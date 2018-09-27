% Fuzzy Systems 2018 - Group 4 - Ser01
% Kosmas Tsiakas 8255
% Classification with TSK models
% Wifi-localization dataset from UCI repository

%% CLEAR
clear all;
close all;
tic
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

NF = [3 9 15 21]; % number of features
NR = [4 8 12 16 20]; % number of rules
error_grid = zeros(length(NF), length(NR));

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

%% GRID SEARCH & 5-fold cross validation
fprintf('\n *** Cross validation \n');

% Keep only the number of features we want and not all of them
% Specify their order and later use the ranks array
[ranks, weights] = relieff(Bank(:, 1:32), Bank(:, 33), 100);

% Check every case for every parameter possible
% For every different case we will save the result in the array parameters
for f = 1 : length(NF)
 
    for r = 1 : length(NR)
        fprintf('\n *** Number of features: %d', NF(f));
        fprintf('\n *** Number of rules: %d \n', NR(r));
     
        % Split the data to make folds and create an array to save the
        % error in each fold
        c = cvpartition(training_data(:, 33), 'KFold', 2);
     
        error = zeros(c.NumTestSets, 1);
     
        % Set the options for genfis
        opt = genfisOptions('SubtractiveClustering');
        % Default options
        opt.ClusterInfluenceRange = 0.5;
        opt.DataScale = 'auto';
        opt.SquashFactor = 1.25;
        opt.AcceptRatio = 0.5;
        opt.RejectRatio = 0.5;
        opt.Verbose = false;
        % C by N array, where C in the number of clusters
        % and N is the number of inputs and outputs
        opt.CustomClusterCenters = zeros(NR(r), NF(f) + 1);
     
        % Generate the FIS
        fprintf('\n *** Generating the FIS\n');
     
        % As input data I give the train_id's that came up with the
        % partitiong and only the most important features
        % As output data is just the last column of the test_data that
        % are left
        init_fis = genfis(training_data(:, ranks(1:NF(f))), training_data(:, 33), opt);
        
        % Plot some input membership functions
figure;
for i = 1 : NF(f)
    [x, mf] = plotmf(init_fis, 'input', i);
    plot(x,mf);
    hold on;
end
title(['Membership functions before training, features:', num2str(NF(f)), 'rules: ' num2str(NR(r))]);
xlabel('x');
ylabel('Degree of membership');
     
        % 5-fold cross validation
        for i = 1 : c.NumTestSets
            fprintf('\n *** Fold #%d\n', i);
         
            train_id = c.training(i);
            test_id = c.test(i);
         
            % Keep separate
            training_data_x = training_data(train_id, ranks(1:NF(f)));
            training_data_y = training_data(train_id, 33);
         
            validation_data_x = training_data(test_id, ranks(1:NF(f)));
            validation_data_y = training_data(test_id, 33);
         
            % Tune the fis
            fprintf('\n *** Tuning the FIS\n');
         
            % Set some options
            % The fis structure already exists
            % set the validation data to avoid overfitting
         
            anfis_opt = anfisOptions('InitialFIS', init_fis, 'EpochNumber', 10, 'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, 'DisplayStepSize', 0, 'DisplayFinalResults', 0, 'ValidationData', [validation_data_x validation_data_y]);
         
            [trn_fis, trainError, stepSize, chkFIS, chkError] = anfis([training_data_x training_data_y], anfis_opt);
         
            % Evaluate the fis
            fprintf('\n *** Evaluating the FIS\n');
         
            % No need to specify specific options for this, keep the defaults
            output = evalfis(validation_data(:, ranks(1:NF(f))), chkFIS);
         
            % Calculate the error
            error(i) = sum((output - validation_data(:, 33)) .^ 2);
        end
     
        cvErr = sum(error) / sum(c.NumTestSets);
        error_grid(f, r) = cvErr
    end
end

%% PLOT THE ERROR
fprintf('The error for diffent values of F and R is:');
error_grid

figure;
surf(error_grid);
title('Error for different number of features and rules');
yticklabels({'3' '9' '15' '21'});
ylabel('Number of features');
xticklabels({'4' '8' '12' '16' '20'});
xlabel('Number of rules');

toc