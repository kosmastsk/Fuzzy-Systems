% Fuzzy Systems 2018 - Group 4 - Ser01
% Kosmas Tsiakas 8255
% Classification with TSK models
% Wifi-localization dataset from UCI repository
tic
%% CLEAR
clear all;
close all;

%% BEGIN
fprintf('\n *** begin %s ***\n\n', mfilename);

%% READ DATA
load wifi-localization.dat
% 2000 instances with 7 features each
data = wifi_localization;
NR = [4 8 12 16]; % number of rules

%% SPLIT DATASET
fprintf('\n *** Dataset splitting\n');

% Keep the data of different outputs to separate arrays
out1 = data(data(:, 8) == 1, :);
out2 = data(data(:, 8) == 2, :);
out3 = data(data(:, 8) == 3, :);
out4 = data(data(:, 8) == 4, :);

% Flags for the index of separating between the sets
split_one = round(0.6 * length(out1));
split_two = round(0.8 * length(out2));

% 60% for training, 20% for validation, 20% for checking
training_data = [out1(1:split_one, :); out2(1:split_one, :); out3(1:split_one, :); out4(1:split_one, :)];
validation_data = [out1(split_one + 1:split_two, :); out2(split_one + 1:split_two, :); out3(split_one + 1:split_two, :); out4(split_one + 1:split_two, :);];
check_data = [out1(split_two + 1:end, :); out2(split_two + 1:end, :); out3(split_two + 1:end, :); out4(split_two + 1:end, :);];

% Shuffle the data
shuffled_data = zeros(size(training_data));
rand_pos = randperm(length(training_data));
for k = 1 : length(training_data)
    shuffled_data(k, :) = training_data(rand_pos(k), :);
end
training_data = shuffled_data;

shuffled_data = zeros(size(validation_data));
rand_pos = randperm(length(validation_data));
% new array
for k = 1 : length(validation_data)
    shuffled_data(k, :) = validation_data(rand_pos(k), :);
end
validation_data = shuffled_data;

shuffled_data = zeros(size(check_data));
rand_pos = randperm(length(check_data));
% new array
for k = 1 : length(check_data)
    shuffled_data(k, :) = check_data(rand_pos(k), :);
end
check_data = shuffled_data;

% Proof that the categories are more or less equally split to each set
count_train = zeros(1, 4);
for i = 1 : length(training_data)
    count_train(training_data(i, 8)) = count_train(training_data(i, 8)) + 1;
end

count_val = zeros(1, 4);
for i = 1 : length(validation_data)
    count_val(validation_data(i, 8)) = count_val(validation_data(i, 8)) + 1;
end

count_chk = zeros(1, 4);
for i = 1 : length(check_data)
    count_chk(check_data(i, 8)) = count_chk(check_data(i, 8)) + 1;
end

%% INITIALIZATIONS
% These arrays will be used for keeping the metrics for the different
% models
error_matrices = cell(1, length(NR)); % Contains 4 7x7 arrays
overall_acc = zeros(1, length(NR));
producers_acc = cell(1, length(NR));
users_acc = cell(1, length(NR));
k = zeros(1, length(NR));

%% TRAIN TSK MODEL

for r = 1 : length(NR)
    fprintf('\n *** Train TSK Model %d \n', r);
    % For matlab > 2017a
    %     % Set the options,
    %     opt = genfisOptions('FCMClustering');
    %     opt.FISType = 'sugeno';
    %     opt.NumClusters = NR(r);
    %     opt.Exponent = 2;
    %     opt.MaxNumIteration = 100;
    %     opt.MinImprovement = 1e-5;
    %     opt.Verbose = false;
    %
    % Generate the FIS
    fprintf('\n *** Generating the FIS\n');
    %     init_fis = genfis(training_data(:, 1:7), training_data(:, 8), opt);
 
    init_fis = genfis3(training_data(:, 1:7), training_data(:, 8), 'sugeno', NR(r));
 
    for m = 1 : length(init_fis.output.mf)
        init_fis.output.mf(m).type = 'constant';
        init_fis.output.mf(m).params = rand(); % range [-5, 5]
    end
 
    % plot input mf
    figure;
    for i = 1 : 7
        [x, mf] = plotmf(init_fis, 'input', i);
        plot(x, mf);
        hold on;
    end
    title(['TSK model ', num2str(r), ': Input MF before training']);
    xlabel('x');
    ylabel('Degree of membership');
    fullFileName = sprintf('%d/input_MF_before_training.png', r);
    saveas(gcf, fullFileName);
 
    % Tune the fis
    fprintf('\n *** Tuning the FIS\n');
    % Set some options
    % The fis structure already exists
    % set the validation data to avoid overfitting
    % display training progress information
    anfis_opt = anfisOptions('InitialFIS', init_fis, 'EpochNumber', 400, 'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, 'ValidationData', validation_data);
 
    [trn_fis, trainError, stepSize, chkFIS, chkError] = anfis(training_data, anfis_opt);
 
    % Evaluate the fis
    fprintf('\n *** Evaluating the FIS\n');
    % No need to specify specific options for this, keep the defaults
    output = evalfis(check_data(:, 1:7), chkFIS);
 
    output = round(output); % Round to the nearest integer to create a constant output for classifying
    %Special cases if the output is 1 or 4 and the round leads to 0 or 5
    output(output < 1) = 1;
    output(output > 4) = 4;
 
    %% METRICS
    N = length(check_data); %total number of classified values compared to truth values
 
    % Error matrix
    error_matrices{r} = confusionmat(check_data(:, 8), output);
    % Columns are truth, rows are predicted values
 
    % Overall accuracy
    error_matrix = error_matrices{r};
    for i = 1 : 4
        overall_acc(r) = overall_acc(r) + error_matrix(i, i);
    end
    overall_acc(r) = overall_acc(r) / N;
 
    % Producer's and user's accuracy
    % probability that a value in a given class was classified correctly
    pa = zeros(1, 4);
    % probability that a value predicted to be in a certain class really is that class
    ua = zeros(1, 4);
 
    for i = 1 : 4
        pa(i) = error_matrix(i, i) / sum(error_matrix(:, i));
        ua(i) = error_matrix(i, i) / sum(error_matrix(i, :));
    end
    producers_acc{r} = pa;
    users_acc{r} = ua;

    % k
    p1 = sum(error_matrix(1, :)) * sum(error_matrix(:, 1)) / N ^ 2;
    p2 = sum(error_matrix(2, :)) * sum(error_matrix(:, 2)) / N ^ 2;
    p3 = sum(error_matrix(3, :)) * sum(error_matrix(:, 3)) / N ^ 2;
    p4 = sum(error_matrix(4, :)) * sum(error_matrix(:, 4)) / N ^ 2;
 
    pe = p1 + p2 + p3 + p4;
 
    k(r) = (overall_acc(r) - pe) / (1 - pe);
 
    % Plot the input membership functions after training
    figure;
    for i = 1 : 7
        [x, mf] = plotmf(chkFIS, 'input', i);
        plot(x, mf);
        hold on;
    end
    title(['TSK model ', num2str(r), ': Input MF after training']);
    xlabel('x');
    ylabel('Degree of membership');
    fullFileName = sprintf('%d/input_MF_after_training.png', r);
    saveas(gcf, fullFileName);
 
    figure;
    plot(1:length(trainError), trainError, 1:length(trainError), chkError);
    title(['TSK model ', r, ': Learning Curve']);
    xlabel('iterations');
    ylabel('Error');
    legend('Training Set', 'Check Set');
    fullFileName = sprintf('%d/learning_curve.png', r);
    saveas(gcf, fullFileName);
 
end

%% PLOT METRICS
figure;
bar(NR(1:length(NR)), overall_acc);
title('Overall accuracy with regards to number of rules');
xlabel('Number of rules');
saveas(gcf, 'overall_accuracy.png');

figure;
bar(NR(1:length(NR)), k);
title('k value with regards to number of rules');
xlabel('Number of rules');
saveas(gcf, 'k_value.png');

save('error_matrices', 'error_matrices');
save('overall_acc', 'overall_acc');
save('k', 'k');
save('producers_acc', 'producers_acc');
save('users_acc', 'users_acc');

save('error_matrices', 'error_matrices');
save('overall_acc', 'overall_acc');
save('producers_acc', 'producers_acc');
save('users_acc', 'users_acc');
save('k', 'k');

toc
%% Elapsed time is 90.226460 seconds.