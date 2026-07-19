%% RF FILTER SYNTHESIZER V2: EXTENDED RANGE [4.5 - 8.5 GHz]
clc; clear; close all;

num_samples = 2500; % Range widened, sample count slightly increased
freq_points = 1001;          
freq = linspace(2e9, 11e9, freq_points); % Axis widened to 2-11 GHz

X_data = zeros(freq_points, 1, 2, num_samples); 
Y_data = zeros(num_samples, 2); 

fprintf('Generating extended range dataset [4.5 - 8.5 GHz]...\n');

for i = 1:num_samples
    f_c_ghz = 4.5 + rand(1)*4.0;    % Range: 4.5 to 8.5 GHz
    b_w_ghz = 0.1 + rand(1)*1.9;    % Range: 0.1 to 2.0 GHz
    order = 5; 
    
    Q = f_c_ghz / b_w_ghz; 
    normalized_freq = (freq/1e9 - f_c_ghz) ./ b_w_ghz; 
    complex_response = 1 ./ (1 + 1i * normalized_freq).^order; 
    
    mag_db = 20 * log10(abs(complex_response) + 1e-6);
    mag_db = mag_db + 0.3 * randn(1, freq_points); 
    
    phase_rad = unwrap(angle(complex_response)); 
    phase_rad = phase_rad + 0.02 * randn(1, freq_points); 
    
    X_data(:, 1, 1, i) = mag_db';
    X_data(:, 1, 2, i) = phase_rad';
    
    % Updated Normalization for new ranges
    fc_norm = (f_c_ghz - 4.5) / 4.0; 
    bw_norm = (b_w_ghz - 0.1) / 1.9;
    Y_data(i, :) = [fc_norm, bw_norm];
end

cv = cvpartition(num_samples, 'HoldOut', 0.2);
XTrain = X_data(:,:,:, training(cv)); YTrain = Y_data(training(cv), :);
XTest  = X_data(:,:,:, test(cv));    YTest  = Y_data(test(cv), :);

layers = [
    imageInputLayer([freq_points 1 2], 'Normalization', 'zscore')
    convolution2dLayer([21 1], 32, 'Padding', 'same')
    batchNormalizationLayer
    leakyReluLayer(0.01)
    maxPooling2dLayer([2 1], 'Stride', [2 1])
    convolution2dLayer([11 1], 64, 'Padding', 'same')
    batchNormalizationLayer
    leakyReluLayer(0.01)
    maxPooling2dLayer([2 1], 'Stride', [2 1])
    convolution2dLayer([5 1], 128, 'Padding', 'same')
    batchNormalizationLayer
    leakyReluLayer(0.01)
    maxPooling2dLayer([2 1], 'Stride', [2 1])
    fullyConnectedLayer(128)
    leakyReluLayer(0.01)
    dropoutLayer(0.1)
    fullyConnectedLayer(2) 
    regressionLayer
];

options = trainingOptions('adam', 'MaxEpochs', 200, 'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.001, 'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 60, 'LearnRateDropFactor', 0.5, ...
    'ValidationData', {XTest, YTest}, 'Plots', 'training-progress', 'Verbose', false);

net = trainNetwork(XTrain, YTrain, layers, options);
save('Kusursuz_AI_Model.mat', 'net');
disp('TRAINING COMPLETE: Extended Model Saved!');