% Fuzzy Systems 2018 - Group 4 - Ser01
% Kosmas Tsiakas 8255
% Classification with TSK models
% Waveform Generation Dataset from UCI repository
tic
%% CLEAR
clear all;
close all;

%% BEGIN
fprintf('\n *** begin %s ***\n\n', mfilename);

%% READ DATA
load waveform.data
% 5000 instances of 40 features each
% 3 classes of waves

%% SPLIT DATASET
fprintf('\n *** Dataset splitting\n');

% Keep the data of different outputs to separate arrays
out1 = waveform(waveform(:, end) == 0, :);
out2 = waveform(waveform(:, end) == 1, :);
out3 = waveform(waveform(:, end) == 2, :);

% Flags for the index of separating between the sets
first_split_one = round(0.6 * length(out1));
second_split_one = round(0.8 * length(out1));

first_split_two = round(0.6 * length(out2));
second_split_two = round(0.8 * length(out2));

first_split_three = round(0.6 * length(out3));
second_split_three = round(0.8 * length(out3));

% 60% for training, 20% for validation, 20% for checking
training_data = [out1(1:first_split_one, :); out2(1:first_split_two, :); out3(1:first_split_three, :)];
validation_data = [out1(first_split_one + 1:second_split_one, :); out2(first_split_two + 1:second_split_two, :); out3(first_split_three + 1:second_split_three, :)];
check_data = [out1(second_split_one + 1:end, :); out2(second_split_two + 1:end, :); out3(second_split_three + 1:end, :)];

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
count_train = zeros(1, 3);
for i = 1 : length(training_data)
    count_train(training_data(i, end) + 1) = count_train(training_data(i, end) + 1) + 1;
end

count_val = zeros(1, 3);
for i = 1 : length(validation_data)
    count_val(validation_data(i, end) + 1) = count_val(validation_data(i, end) + 1) + 1;
end

count_chk = zeros(1, 3);
for i = 1 : length(check_data)
    count_chk(check_data(i, end) + 1) = count_chk(check_data(i, end) + 1) + 1;
end

% Keep only the number of features we want and not all of them
% Specify their order and later use the ranks array
[ranks, weights] = relieff(waveform(:, 1:40), waveform(:, end), 100);

%% FINAL TSK MODEL
fprintf('\n *** TSK Model with 15 features and 7 rules - Substractive Clustering\n');

f = 15;
radii = 0.8;

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

training_data_x = training_data(:, ranks(1:f));
training_data_y = training_data(:, end);

validation_data_x = validation_data(:, ranks(1:f));
validation_data_y = validation_data(:, end);

check_data_x = check_data(:, ranks(1:f));
check_data_y = check_data(:, end);

%% TRAIN TSK MODEL

%% MODEL WITH 15 FEATURES AND 7 RULES

% Generate the FIS
fprintf('\n *** Generating the FIS\n');

% As input data I give the train_id's that came up with the
% partitioning and only the most important features
% As output data is just the last column of the test_data that
% are left
% init_fis = genfis(training_data_x, training_data_y, opt);

init_fis = genfis2(training_data(:, ranks(1:f)),training_data(:, end), radii) ;
rules = length(init_fis.rule);

% Plot some input membership functions
figure;
for i = 1 : f
    [x, mf] = plotmf(init_fis, 'input', i);
    plot(x, mf);
    hold on;
end
title('Membership functions before training');
xlabel('x');
ylabel('Degree of membership');
saveas(gcf, 'Final_TSK_model/mf_before_training.png');

figure;
[x, mf] = plotmf(init_fis, 'input', 1);
subplot(2, 2, 1);
plot(x, mf);
xlabel('input 1');

[x, mf] = plotmf(init_fis, 'input', 2);
subplot(2, 2, 2);
plot(x, mf);
xlabel('input 10');

[x, mf] = plotmf(init_fis, 'input', 3);
subplot(2, 2, 3);
plot(x, mf);
xlabel('input 14');

[x, mf] = plotmf(init_fis, 'input', 4);
subplot(2, 2, 4);
plot(x, mf);
xlabel('input 20');

suptitle('Final TSK model : some membership functions before training');
saveas(gcf, 'Final_TSK_model/some_mf_before_training.png');

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
output = evalfis(check_data_x, chkFIS);

error = sum((output - check_data(:, end)) .^ 2); %mse

output = round(output); % Round to the nearest integer to create a constant output for classifying
%Special cases if the output is 1 or 4 and the round leads to 0 or 5
output(output < 0) = 0;
output(output > 2) = 2;

%% METRICS
N = length(check_data); %total number of classified values compared to truth values

% Error matrix
error_matrix = confusionmat(check_data_y, output)
% Columns are truth, rows are predicted values

% Overall accuracy
overall_acc = 0;
for i = 1 : 3
    overall_acc = overall_acc + error_matrix(i, i);
end
overall_acc = overall_acc / N

% Producer's and user's accuracyρακολούθηση όλων των ομιλιών, γεύμα και ροφήματα, μπλουζάκι και άλλα αναμνηστικά δώρα και (π
% probability that a value in a given class was classified correctly
pa = zeros(1, 3);
% probability that a value predicted to be in a certain class really is that class
ua = zeros(1, 3);

for i = 1 : 3
    pa(i) = error_matrix(i, i) / sum(error_matrix(:, i));
    ua(i) = error_matrix(i, i) / sum(error_matrix(i, :));
end

% k
p1 = sum(error_matrix(1, :)) * sum(error_matrix(:, 1)) / N ^ 2;
p2 = sum(error_matrix(2, :)) * sum(error_matrix(:, 2)) / N ^ 2;
p3 = sum(error_matrix(3, :)) * sum(error_matrix(:, 3)) / N ^ 2;
 
pe = p1 + p2 + p3;
 
k = (overall_acc - pe) / (1 - pe);

% Plot the metrics
% Predictions vs real values
figure;
plot(1:length(check_data_x), check_data_y, '*r', 1:length(check_data_y), output, '.b');
title('Final TSK model : Model vs Reference Output');
legend('Reference Outputs', 'Model Outputs');
saveas(gcf, 'Final_TSK_model/output.png')

% Plot the input membership functions after training
figure;
for i = 1 : f
    [x, mf] = plotmf(chkFIS, 'input', i);
    plot(x, mf);
    hold on;
end
title('Final TSK model : Input MF after training');
xlabel('x');
ylabel('Degree of membership');
fullFileName = sprintf('Final_TSK_model/input_MF_after_training.png');
saveas(gcf, fullFileName);

figure;
[x, mf] = plotmf(chkFIS, 'input', 1);
subplot(2, 2, 1);
plot(x, mf);
xlabel('input 1');

[x, mf] = plotmf(chkFIS, 'input', 2);
subplot(2, 2, 2);
plot(x, mf);
xlabel('input 10');

[x, mf] = plotmf(chkFIS, 'input', 3);
subplot(2, 2, 3);
plot(x, mf);
xlabel('input 14');

[x, mf] = plotmf(chkFIS, 'input', 4);
subplot(2, 2, 4);
plot(x, mf);
xlabel('input 20');

suptitle('Final TSK model : some membership functions after training');
saveas(gcf, 'Final_TSK_model/some_mf_after_training.png');

% Learning curve
figure;
plot(1:length(trainError), trainError, 1:length(trainError), chkError);
title('Final TSK model : Learning Curve');
xlabel('iterations');
ylabel('Error');
legend('Training Set', 'Check Set');
fullFileName = sprintf('Final_TSK_model/learning_curve.png');
saveas(gcf, fullFileName);

toc

%% Elapsed time is 198.104307 seconds.