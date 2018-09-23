% Fuzzy Systems 2018 - Group 3 - Ser06
% Kosmas Tsiakas 8255
% Regression with TSK models 
% Bank dataset from delve repository

%% CLEAR
clear all;
close all;
tic
%% BEGIN
fprintf('\n *** begin %s ***\n\n',mfilename);

%% READ DATA
load Bank.data 

NF = [3 9 15 21]; % number of features
NR = [4 8 12 16 20]; % number of rules
error_grid = zeros(length(NF), length(NR));

%% SPLIT DATASET
fprintf('\n *** Dataset splitting\n');

training_data = Bank(1 : round(0.6*8192), :); % 60% of the dataset is for training
validation_data = Bank(round(0.6*8192)+1 : round(0.8 * 8192), :); % 20% is for evaluation
check_data = Bank(round(0.8*8192)+1 : end, :); % 20% is for testing

%% GRID SEARCH & 5-fold cross validation
fprintf('\n *** Cross validation \n');

% Check every case for every parameter possible
% For every different case we will save the result in the array parameters
for f = 1 : length(NF)
    for r = 1 : length(NR)
        fprintf('\n *** Number of features: %d', NF(f));
        fprintf('\n *** Number of rules: %d \n', NR(r));
        
        % Keep only the number of features we want and not all of them
        [ranks,weights] = relieff(Bank(:,1:32), Bank(:,33), NF(f));
        
        % Split the data to make folds and create an array to save the
        % error in each fold
        c = cvpartition(Bank(:,33),'KFold',5);
        error = zeros(c.NumTestSets,1);
        
        % 5-fold cross validation
        for i = 1 : c.NumTestSets
            fprintf('\n *** Fold #%d\n', i);
    
            train_id = c.training(i);
            test_id = c.test(i);
            
            training_data_x = Bank(train_id, ranks(1:NF(f)));
            training_data_y = Bank(train_id, 33);
            
            % Keep the 20% of the whole dataset for validation
            validation_data_x = training_data_x(round(0.66*length(training_data_x))+1 : end, :);
            validation_data_y = training_data_y(round(0.66*length(training_data_y))+1 : end, :);
            
            training_data_x = training_data_x(1 : round(0.66*length(training_data_x)), :);
            training_data_y = training_data_y(1 : round(0.66*length(training_data_y)), :);
           
            test_data_x = Bank(test_id, ranks(1:NF(f)));
            test_data_y = Bank(test_id, 33);
            
            % Set the options, 
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
            init_fis = genfis(training_data_x, training_data_y, opt);
    
            showrule(init_fis);
            
            % Tune the fis
            fprintf('\n *** Tuning the FIS\n');
            
            % Set some options
            % The fis structure already exists
            % set the validation data to avoid overfitting
            
            anfis_opt = anfisOptions('InitialFIS', init_fis, 'EpochNumber', 400, 'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, 'DisplayStepSize', 0, 'DisplayFinalResults', 0, 'ValidationData', [validation_data_x validation_data_y]);

            [trn_fis, trainError, stepSize, chkFIS, chkError] = anfis([training_data_x training_data_y], anfis_opt);
            
            % Evaluate the fis
            fprintf('\n *** Evaluating the FIS\n');
    
            % No need to specify specific options for this, keep the defaults
            output = evalfis(test_data_x,chkFIS);
            
            % Calculate the error
            error(i) = sum( (output - test_data_y).^2);
            
       end
        
        cvErr = sum(error)/sum(c.TestSize);
        error_grid(f, r) = cvErr;
    end 
end

%% PLOT ERROR
figure;
surf(error_grid);
colorbar;
title('Error for different values of features and rules');


toc