close all, clear all, clc; 
%% 
load('Features.mat')
load('Label.mat')
load('subject.mat')
%The features and activity that are choosen
nr1= 6;
nr2 = 2;
labelnr = 2;

%randomizing
ix = randperm(length(label));
features = features(ix,:);
label = label(ix);

featuresLin=featureNormalize(features);

%% Excercise 1
feat_tr = featuresLin(1:round((length(label)*0.4)),:); 
feat_val = featuresLin(round((length(label)*0.4))+1:round((length(label)*0.7)),:); 
feat_test = featuresLin(round((length(label)*0.7))+1:length(label),:); 

label_tr = label(1:round((length(label)*0.4)));
label_val= label(round((length(label)*0.4))+1:round((length(label)*0.7))); 
label_test = label(round((length(label)*0.7))+1:length(label)); 

% 1 vs all classification
activity = label == labelnr;        %Sitting down (label = 6)
activity = double(activity);
plotlabel = label;

% All the features:

gplotmatrix(features,features,activity);
% The selected features
figure;
gplotmatrix(features(:,nr1),features(:,nr2),activity);

%% Exercise 2 
%2.2

% Normalize the training dataset
X_Norm = ones(length(label_tr),2);
X_Norm(:,1) = feat_tr(:,nr1);
X_Norm(:,2) = feat_tr(:,nr2);
%[X_Norm, mu, sigma] = featureNormalize(X_Norm);
%X_Norm =featureNormalize(X_Norm); %Normalization of the two features

% create X and y-values for the training set (normalized)
X_tr = ones(length(label_tr),3);
X_tr(:,2) = X_Norm(:,1);
X_tr(:,3) = X_Norm(:,2);
y_tr = activity(1:round((length(label)*0.4)),:); 

%Finding optimal theta
lambda = 0;
[theta, cost, exit_flag] = training(X_tr, y_tr, lambda);

% Normalize the test data:
% X_Norm_test = ones(length(label_test),2);
% X_Norm_test(:,1) = feat_test(:,nr1);
% X_Norm_test(:,2) = feat_test(:,nr2);
% X_norm_test = bsxfun(@minus, X_Norm_test, mu);
% X_norm_test = bsxfun(@rdivide, X_norm_test, sigma);
% X_norm_test = [ones(size(X_norm_test, 1), 1), X_norm_test];         % Add Ones

% Normalize the validation data:
X_Norm_val = ones(length(label_val),2);
X_Norm_val(:,1) = feat_val(:,nr1);
X_Norm_val(:,2) = feat_val(:,nr2);
% X_norm_val = bsxfun(@minus, X_Norm_val, mu);
% X_norm_val = bsxfun(@rdivide, X_norm_val, sigma);
X_norm_val = [ones(size(X_Norm_val, 1), 1), X_Norm_val];           % Add Ones

% y values for training and validation:
y_val = activity(round((length(label)*0.4))+1:round((length(label)*0.7)));
y_test = activity(round((length(label)*0.7))+1:length(label));

%Training score before and after training
initial_theta = zeros(size(X_tr, 2), 1); 
score_before_tr = F1_score(X_tr,initial_theta,y_tr);
score_after_tr = F1_score(X_tr,theta,y_tr);

%Validation score before and after training
score_before_val = F1_score(X_norm_val,initial_theta,y_val);
score_after_val = F1_score(X_norm_val,theta,y_val);

%Plotting the linear decision-boundary
plotDecisionBoundary(theta,X_norm_val,y_val)
%plotDecisionBoundary(theta,X_tr,y_tr)

%% 

%2.3 
feat_tr = features(1:round((length(label)*0.4)),:); 
feat_val = features(round((length(label)*0.4))+1:round((length(label)*0.7)),:); 
feat_test = features(round((length(label)*0.7))+1:length(label),:); 

% Adding polonial features with degree p
p = 6;
X_poly = mapFeature(feat_tr(:,nr1),feat_tr(:,nr2),p);
X_poly_test = mapFeature(feat_test(:,nr1),feat_test(:,nr2),p);
X_poly_val = mapFeature(feat_val(:,nr1),feat_val(:,nr2),p);

% Generate a vector of different lambdas and calculate the f1 score for
% each lambda:

[lambda_vec, f1_train, f1_val] = ...
    validationCurve(X_poly, y_tr, X_poly_val, y_val);

% Plot f1 score as a function of lambda for training and validation
figure;
semilogx(lambda_vec, f1_train, lambda_vec, f1_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('F1 score');
title('Two feat CV poly of 6th power');

% fprintf('lambda\t\tTrain Error\tValidation Error\n');
% for i = 1:length(lambda_vec)
% 	fprintf(' %f\t%f\t%f\n', ...
%             lambda_vec(i), f1_train(i), f1_val(i));
% end

% Find the optimal lambda:
[val, idx] = max(f1_train);
lambda = lambda_vec(idx);

% Train
initial_theta = zeros(28,1);
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X_poly, y_tr, lambda)), initial_theta, options);
plotDecisionBoundary(theta, X_poly_val, y_val);
title(sprintf('CV: lambda = %g and F1 score = %d', lambda, F1_score(X_poly_val,theta,y_val)))

%% 

%2.4
% a) - Linear
X8_norm = featureNormalize(feat_tr);
X8_norm = [ones(size(X8_norm, 1), 1), X8_norm];
X8_val = featureNormalize(feat_val);
X8_val = [ones(size(X8_val, 1), 1), X8_val];

lambda = 0;
[theta, cost, exit_flag] = training(X8_norm, y_tr, lambda);

[lambda_vec, f1_train, f1_val] = ...
    validationCurve(X8_norm, y_tr, X8_val, y_val);

figure;
semilogx(lambda_vec, f1_train, lambda_vec, f1_val);
title(['Eight feat linear traininga CV'])
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('F1 Score');

% b) Non-linear

out = ones(size(feat_tr(:,1)));
X8_poly = x2fx(feat_tr,'quadratic');
X8_poly_val = x2fx(feat_val,'quadratic');

[lambda_vec, f1_train, f1_val] = ...
    validationCurve(X8_poly, y_tr, X8_poly_val, y_val);

figure;
semilogx(lambda_vec, f1_train, lambda_vec, f1_val);
title(['Eight poly training and CV'])
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('F1');

%%
% Adding training examples

tic

l = 100;        % Step size
t = 4000;       % Number of extra training examples
training = 1:l:t;
f1_score_tr_few = zeros(t/l,1);
f1_score_val_few =  zeros(t/l,1);
count  = 1;

for i=training
    X8_poly_few = X8_poly(1:i,:);
    y_tr_few = y_tr(1:i);
    
    [lambda_vec_few, f1_train_few, f1_val_few] = ...
    validationCurve(X8_poly_few, y_tr_few, X8_poly_val, y_val);
    
    f1_score_tr_few(count) = max(f1_train_few);
    f1_score_val_few(count) = max(f1_val_few);
    count = count+1
end
toc

% Plot the training and CV for different training examples

plot(training,f1_score_tr_few(1:40))
hold on
plot(training, f1_score_val_few)
axis([0 4000 0 1])
legend('Training','CV')
title('Adding more training examples')
xlabel('Number of extra training examples');
ylabel('F1');







