% Fuzzy Systems 2018 - Group 3 - Ser06
% Kosmas Tsiakas 8255
% Regression with TSK models
% Bank dataset from delve repository

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
% values for radii
RADII = [0.1 0.3 0.5 0.7 0.9 ; % f = 3
         0.5 0.6 0.7 0.9 1.0 ; % f = 9
         0.5 0.6 0.7 0.8 1.0 ; % f = 15
         0.6 0.7 0.8 0.9 1.0]; % f = 21

error_grid = zeros(length(NF), length(RADII));
rule_grid = zeros(length(NF), length(RADII));

%% SPLIT DATASET
fprintf('\n *** Dataset splitting\n');

training_data = shuffledBank(1 : round(0.6 * 8192), :); % 60% of the dataset is for training
validation_data = shuffledBank(round(0.6 * 8192) + 1 : round(0.8 * 8192), :); % 20% is for evaluation
check_data = shuffledBank(round(0.8 * 8192) + 1 : end, :); % 20% is for testing

%% NORMALIZE DATA
% Normalize each set differently so that they are separated through the
% whole process
for i = 1 : size(training_data, 2) - 1
    training_data_min = min(training_data(:, i));
    training_data_max = max(training_data(:, i));
 
    training_data(:, i) = (training_data(:, i) - training_data_min) / (training_data_max - training_data_min); % Scaled to [0, 1]
    training_data(:, i) = training_data(:, i) * 2 - 1;
 
    validation_data(:, i) = (validation_data(:, i) - training_data_min) / (training_data_max - training_data_min); % Scaled to [0, 1]
    validation_data(:, i) = validation_data(:, i) * 2 - 1;
 
    check_data(:, i) = (check_data(:, i) - training_data_min) / (training_data_max - training_data_min); % Scaled to [0, 1]
    check_data(:, i) = check_data(:, i) * 2 - 1;
end

%% GRID SEARCH & 5-fold cross validation
fprintf('\n *** Cross validation \n');

% Keep only the number of features we want and not all of them
% Specify their order and later use the ranks array
[ranks, weights] = relieff(Bank(:, 1:end - 1), Bank(:, end), 100);

% Check every case for every parameter possible
% For every different case we will save the result in the array parameters
for f = 1 : length(NF)
 
    for r = 1 : length(RADII)
        fprintf('\n *** Number of features: %d', NF(f));
        fprintf('\n *** Radii value: %d \n', RADII(f, r));
     
        % Split the data to make folds and create an array to save the
        % error in each fold
        c = cvpartition(training_data(:, end), 'KFold', 5);
        error = zeros(c.NumTestSets, 1);
     
        % Generate the FIS
        fprintf('\n *** Generating the FIS\n');
     
        % As input data I give the train_id's that came up with the
        % partitiong and only the most important features
        % As output data is just the last column of the test_data that
        % are left
        init_fis = genfis2(training_data(:, ranks(1:NF(f))), training_data(:, end), RADII(f, r));
        rule_grid(f, r) = length(init_fis.rule)
        if (rule_grid(f, r) == 1 || rule_grid(f,r) > 100) % if there is only one rule we cannot create a fis, so continue to next values
            continue; % or more than 100, continue, for speed reason
        end
        % 5-fold cross validation
        for i = 1 : c.NumTestSets
            fprintf('\n *** Fold #%d\n', i);
         
            train_id = c.training(i);
            test_id = c.test(i);
         
            % Keep separate
            training_data_x = training_data(train_id, ranks(1:NF(f)));
            training_data_y = training_data(train_id, end);
         
            validation_data_x = training_data(test_id, ranks(1:NF(f)));
            validation_data_y = training_data(test_id, end);
         
            % Tune the fis
            fprintf('\n *** Tuning the FIS\n');
         
            % Set some options
            % The fis structure already exists
            % set the validation data to avoid overfitting
         
            anfis_opt = anfisOptions('InitialFIS', init_fis, 'EpochNumber', 50, 'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, 'DisplayStepSize', 0, 'DisplayFinalResults', 0, 'ValidationData', [validation_data_x validation_data_y]);
         
            [trn_fis, trainError, stepSize, init_fis, chkError] = anfis([training_data_x training_data_y], anfis_opt);
         
            % Evaluate the fis
            fprintf('\n *** Evaluating the FIS\n');
         
            % No need to specify specific options for this, keep the defaults
            output = evalfis(validation_data(:, ranks(1:NF(f))), init_fis);
         
            % Calculate the error
            error(i) = sum((output - validation_data(:, end)) .^ 2);
        end
     
        cvErr = sum(error) / c.NumTestSets;
        error_grid(f, r) = cvErr / length(output)
    end
end

%% PLOT THE ERROR
fprintf('The error for diffent values of F and Radii is:');
error_grid
% save('error_grid', 'error_grid');

fprintf('The number of rules created for diffent values of F and Radii is:');
rule_grid
% save('rule_grid', 'rule_grid');


%% PLOT
figure;
suptitle('Error for different number of features and radii values');

subplot(2,2,1);
bar(error_grid(1,:))
xlabel('radii value');
ylabel('Mean Square Error');
xticklabels({'0.1','0.3','0.5','0.7','0.9'});
legend('3 features')

subplot(2,2,2);
bar(error_grid(2,:));
xlabel('radii value');
ylabel('Mean Square Error');
xticklabels({'0.5','0.6','0.7','0.9','1'});
legend('9 features')

subplot(2,2,3);
bar(error_grid(3,:));
xlabel('radii value');
ylabel('Mean Square Error');
xticklabels({'0.5','0.6','0.7','0.8','1'});
legend('15 features')

subplot(2,2,4);
bar(error_grid(4,:));
xlabel('radii value');
ylabel('Mean Square Error');
xticklabels({'0.6','0.7','0.8','0.9','1'});
legend('21 features')
saveas(gcf, 'error_grid_wrg_f_r.png');

figure;
bar3(error_grid);
ylabel('Number of feature');
yticklabels({'3','9','15','21'});
xlabel('Radii values');
xticklabels({'1st','2nd','3rd','4th'});
zlabel('Mean square error');
title('Error for different number of features and radii');
saveas(gcf, 'error_wrt_f_r.png');

figure;
bar3(rule_grid);
ylabel('Number of features');
yticklabels({'3','9','15','21'});
xlabel('Radii values');
xticklabels({'1st','2nd','3rd','4th'});
zlabel('Number of rules created');
title('Rules created for different number of features and radii');
saveas(gcf, 'rules_wrt_f_r.png');

toc