load('C:\Users\Gustav\Desktop\Plugg Leuven\Machine Learning\Lab1\Master-LR\Dataset\Features.mat')
load('C:\Users\Gustav\Desktop\Plugg Leuven\Machine Learning\Lab1\Master-LR\Dataset\Label.mat')
load('C:\Users\Gustav\Desktop\Plugg Leuven\Machine Learning\Lab1\Master-LR\Dataset\subject.mat')
%The feature that is choosen
nr1= 6;
nr2 = 2;

%randomizing
ix = randperm(length(label));
features = features(ix,:);
label = label(ix);

%Excercise 1
feat_tr = features(1:round((length(label)*0.4)),:); 
feat_val = features(round((length(label)*0.4))+1:round((length(label)*0.7)),:); 
feat_test = features(round((length(label)*0.7))+1:length(label),:); 

label_tr = label(1:round((length(label)*0.4)));
label_val= label(round((length(label)*0.4))+1:round((length(label)*0.7))); 
label_test = label(round((length(label)*0.7))+1:length(label)); 

% 1 vs all classification
activity = label == 6; %Sitting down (label = 6)
activity = double(activity);
plotlabel = label;

%figure;
%gplotmatrix(features(:,nr1),features(:,nr2),activity);
%Feature nr

X_Norm = ones(length(label_tr),2);
X_Norm(:,1) = feat_tr(:,nr1);
X_Norm(:,2) = feat_tr(:,nr2);
[X_Norm, mu, sigma] = featureNormalize(X_Norm)
X_Norm =featureNormalize(X_Norm); %Normalization of the two features

% X and y-values for the training set
X_tr = ones(length(label_tr),3);
X_tr(:,2) = X_Norm(:,1);
X_tr(:,3) = X_Norm(:,2);
y_tr = activity(1:round((length(label)*0.4)),:); 

%Finding optimal theta
lambda = 0;
[theta, cost, exit_flag] = training(X_tr, y_tr, lambda);

%NYTT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
X_Norm_test = ones(length(label_test),2);
X_Norm_test(:,1) = feat_test(:,nr1);
X_Norm_test(:,2) = feat_test(:,nr2);

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = bsxfun(@minus, X_Norm_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

X_Norm_val = ones(length(label_val),2);
X_Norm_val(:,1) = feat_val(:,nr1);
X_Norm_val(:,2) = feat_val(:,nr2);

% Map X_poly_val and normalize (using mu and sigma)
X_norm_val = bsxfun(@minus, X_Norm_val, mu);
X_norm_val = bsxfun(@rdivide, X_norm_val, sigma);
X_norm_val = [ones(size(X_norm_val, 1), 1), X_norm_val];           % Add Ones
%NYTT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

%Detta blev överflödigt nu
%X and y- values for the validation set
% X1_val = feat_val(:,nr1);
% X2_val = feat_val(:,nr2);
% X_val = ones(length(label_val),3);
% X_val(:,2) = X1_val;
% X_val(:,3) = X2_val;
 y_val = activity(round((length(label)*0.4))+1:round((length(label)*0.7)));

%Training score
initial_theta = zeros(size(X_tr, 2), 1); 
score_before_tr = F1_score(X_tr,initial_theta,y_tr);
score_after_tr = F1_score(X_tr,theta,y_tr);
%Validation score
score_before_val = F1_score(X_norm_val,initial_theta,y_val);
score_after_val = F1_score(X_norm_val,theta,y_val);

%Plotting the decision-boundary
%plotDecisionBoundary(theta,X_tr,y_val)

%fprintf('Program paused. Press enter to continue.\n');
%pause;


%2.3 
%adding polonial features
X_pol = mapFeature(feat_tr(:,nr1),feat_tr(:,nr2),6);
%lambda_pol = (3^(-10)):.2:(3^(10));
%!!!!!!!!!!!!!!!!!!!!!!!!!!!NEDAN ÄR NYTT!!!!!

p = 6;
X_poly = mapFeature(feat_tr(:,nr1),feat_tr(:,nr2),p);
X_poly(:,1) = [];
[X_poly, mu, sigma] = featureNormalize(X_poly) ; % Normalize
X_poly = [ones(length(label_tr), 1), X_poly]; % Add Ones

% Map X_poly_test and normalize (using mu and sigma)

X_poly_test = mapFeature(feat_test(:,nr1),feat_test(:,nr2),p);
X_poly_test(:,1) = [];
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = mapFeature(feat_val(:,nr1),feat_val(:,nr2),p);
X_poly_val(:,1) = [];
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma); 
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];    %Add ones 

[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y_tr, X_poly_val, y_val);

close all;
semilogx(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end
lambda=lambda_vec(length(lambda_vec));
initial_theta = zeros(28,1);
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X_poly, y_tr, lambda)), initial_theta, options);

%fprintf('Program paused. Press enter to continue.\n');
%pause;
plotDecisionBoundary(theta, X_poly, y_tr);
hold on;
title(sprintf('lambda = %g', lambda))

