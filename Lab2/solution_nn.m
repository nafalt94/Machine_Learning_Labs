%% Preparation 1
close all
clear
load('Train_raw_data.mat');
load('Validation_raw_data.mat');
load('Test_raw_data.mat');
load('Labels.mat');

labels = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes", "unknown"];

% listen to and plot the audio
soundsc(Train_raw_data(:, 1001), fs);
figure; plot(1:size(Train_raw_data, 1), Train_raw_data(:, 1001));

%% Normalisation in time domain (optional)
%Train_raw_data = zscore(Train_raw_data, [],1);
%Validation_raw_data = zscore(Validation_raw_data, [],1);
%Test_raw_data = zscore(Test_raw_data, [],1);


%% Feature Extraction
% feature parameteres
segmentDuration = 1;
frameDuration = 0.025;
hopDuration = 0.010;
numBands = 40;

% extracting features
epsil = 1e-6;
XTrain = speechSpectrograms(Train_raw_data,fs,segmentDuration,frameDuration,hopDuration,numBands);
XTrain = log10(XTrain + epsil);

XValidation = speechSpectrograms(Validation_raw_data,fs,segmentDuration,frameDuration,hopDuration,numBands);
XValidation = log10(XValidation + epsil);

XTest = speechSpectrograms(Test_raw_data,fs,segmentDuration,frameDuration,hopDuration,numBands);
XTest = log10(XTest + epsil);

% plot the feature
figure; 
idx = 1201;
soundsc(Test_raw_data(:, idx), fs);
subplot(2,1,1); plot(1:size(Test_raw_data, 1), Test_raw_data(:, idx)); title(["Raw data - Class:" labels(YTest(idx))]);
subplot(2,1,2); image(flip(XTest(:,:,1,idx), 1),'CdataMapping','scaled'); title("Spectrogram");

%% Reshape the features into vectors and each row corresponds to one data

Xtr_reshape = reshape(XTrain,3920,21789)';
Xval_reshape = reshape(XValidation,3920,2975)';
Xtest_reshape = reshape(XTest,3920,2984)';
                          
%% Design the ANN model
 
input_layer_size  = 3920;  % 40x98
hidden_layer_size = 256;   % 256 hidden units
num_labels = 11;          % 10 labels, from 1 to 10  

% The matrices Theta1 and Theta2 will now be in your workspace
% Theta1 has size 25 x 401 - In this lab it has 25 X 3921
% Theta2 has size 10 x 26 - In this lab it has 11 X 26

% the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.
epsilon_init = 0.12;
Theta1 = rand(hidden_layer_size,(input_layer_size+1)) * (2 * epsilon_init) - epsilon_init;
Theta2 = rand(11,(hidden_layer_size+1)) * (2 * epsilon_init) - epsilon_init;
%Theta3 = rand(1,11) * (2 * epsilon_init) - epsilon_init;

nn_params = [Theta1(:) ; Theta2(:)];

%% Training the ANN model
% Weight regularization parameter (we set this to 0 here).
lambda = 0;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, Xtr_reshape, YTrain, lambda);
               
         
%  Once your cost function implementation is correct, you should now
%  continue to implement the regularization with the cost.
%

fprintf('\nChecking Cost Function (w/ Regularization) ... \n')

% Weight regularization parameter (we set this to 1 here).
lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, Xtr_reshape, YTrain, lambda);
               
fprintf(['Cost: %f '...
         '\n\n'], J);

     %% ================ Part 5: Sigmoid Gradient  ================
%  Before you start implementing the neural network, you will first
%  implement the gradient for the sigmoid function. You should complete the
%  code in the sigmoidGradient.m file.
%

fprintf('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient([-1 -0.5 0 0.5 1]);

fprintf('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =============== Part 7: Implement Backpropagation ===============
%  Once your cost matches up with ours, you should proceed to implement the
%  backpropagation algorithm for the neural network. You should add to the
%  code you've written in nnCostFunction.m to return the partial
%  derivatives of the parameters.
%
fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% =============== Part 8: Implement Regularization ===============
%  Once your backpropagation implementation is correct, you should now
%  continue to implement the regularization with the cost and gradient.
%

fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);

% Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' ...
         '\n(for lambda = 3, this value should be about 0.576051)\n\n'], lambda, debug_J);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =================== Part 8: Training NN ===================

fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 2);
%  You should also try different values of lambda
lambda = [0.0001 0.001 0.01 0.1 1 10 100];
%Jvec = zeros(7,1);
%theta1vec=[];
%theta2vec=[];

 M = 4;                     % M specifies maximum number of workers
 tic
parfor (i = 1:7,M)
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, Xtr_reshape, YTrain, lambda(i));

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
cost
Jvec(i) = min(cost);
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
Theta2
theta1vec(i,:,:) = Theta1;
theta2vec(i,:,:) = Theta2;
        
end
toc

%% ================= Part 10: Implement Predict =================
predvec=[];



for i = 1:size(Jvec)
    dummy1 = reshape(theta1vec(i,:,:),256,3921);
    dummy2 = reshape(theta2vec(i,:,:),11,257);
   predvec(i,:)=  predict(dummy1, dummy2, Xval_reshape);
   predvec_tr(i,:) =  predict(dummy1, dummy2, Xtr_reshape);
    
end
%pred = predict(theta1vec(1,:,:), theta2vec(1,:,:), Xval_reshape);
valAcc= [];
trAcc=[];
for i = 1:size(Jvec)
%fprintf('\nTraining Set Accuracy: %f\n', mean(double(predvec(i) == YValidation)) * 100);
valAcc(i) =  mean(double(predvec(i) == YValidation)) * 100; 
trAcc(i) = mean(double(predvec_tr(i) == YTrain)) * 100;
end
plot(lambda,valAcc)
     
%% Plotting training and validation accuracy for diferent lambda

 

%% Try the trained model in real-life
%    run real_time_command_recognition_app
