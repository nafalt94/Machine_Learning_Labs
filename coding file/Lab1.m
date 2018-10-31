load('C:\Users\Gustav\Desktop\Plugg Leuven\Machine Learning\Lab1\Master-LR\Dataset\Features.mat')
load('C:\Users\Gustav\Desktop\Plugg Leuven\Machine Learning\Lab1\Master-LR\Dataset\Label.mat')
load('C:\Users\Gustav\Desktop\Plugg Leuven\Machine Learning\Lab1\Master-LR\Dataset\subject.mat')
%The feature that is choosen
nr1= 6;
nr2 = 1;

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
activity = label == 6; %Sitting down
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

%2.3 
%adding polonial features
X_pol = mapFeature(feat_tr(:,nr1),feat_tr(:,nr2),6);
%lambda_pol = (3^(-10)):.2:(3^(10));

lambda_pol(1) = 3^(-10);
i = 1;
while(lambda_pol(i)<(3^(10)))
   i=i+1;
    lambda_pol(i) = lambda_pol(i-1)*2; 
end

theta_vector = zeros(length(lambda_pol),3);
score_vector_tr = zeros(length(lambda_pol),1);
score_vector_val = zeros(length(lambda_pol),1);

for i=1:length(lambda_pol)
[theta, cost, exit_flag] = training(X_tr, y_tr, lambda_pol(i)); 
theta_vector(i,:) = theta;
score_vector_tr(i) = F1_score(X_tr,theta,y_tr);
score_vector_val(i) = F1_score(X_norm_val,theta,y_val);
end

%Plotting lambda and F1-score for training and validation
%figure
%hold on
%plot(lambda_pol,score_vector_val)
%plot(lambda_pol,score_vector_tr)
%axis([0 700 0 1])
%legend('Cross validation','Training')
%When lambda increases --> High bias

%Hittar vilken lambda som är bäst
[val, idx] = max(score_vector_val);
%plotDecisionBoundary(theta_vector(idx,:),X_val,y_val)

% going from 8 features to quad ratic representation
% använd x2fx



