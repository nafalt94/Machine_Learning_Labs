load('C:\Users\Gustav\Desktop\Plugg Leuven\Machine Learning\Lab1\Master-LR\Dataset\Features.mat')
load('C:\Users\Gustav\Desktop\Plugg Leuven\Machine Learning\Lab1\Master-LR\Dataset\Label.mat')
load('C:\Users\Gustav\Desktop\Plugg Leuven\Machine Learning\Lab1\Master-LR\Dataset\subject.mat')
%The feature that is choosen
nr= 6;

ix = randperm(length(label));

%randomizing
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
activity = label == nr;
activity = double(activity);
plotlabel = label;

%figure;
%gplotmatrix(features(:,5),features(:,2),activity);
%Feature nr

% X and y-values for the training set
X1_tr = feat_tr(:,nr);
X2_tr = feat_tr(:,2);
X_tr = ones(length(label_tr),3);
X_tr(:,2) = X1_tr;
X_tr(:,3) = X2_tr;
y_tr = activity(1:round((length(label)*0.4)),:); 
%X and y- values for the validation set
X1_val = feat_val(:,nr);
X2_val = feat_val(:,2);
X_val = ones(length(label_val),3);
X_val(:,2) = X1_val;
X_val(:,3) = X2_val;
y_val = activity(round((length(label)*0.4))+1:round((length(label)*0.7)));

lambda = 0;
%initial_theta = zeros(n + 1, 1);
[theta, cost, exit_flag] = training(X_tr, y_tr, lambda);

%N�r vi r�knar ut F1 score ska vi r�kna ut F1 score f�r b�de training set
%och validation f�r detta theta. N�r vi i senare del tunar lambda s� ska vi
%anv�nda en hypotesen f�r att testa p� test-setet.

%Training score
score_before_tr = F1_score(X_tr,initial_theta,y_tr);
score_after_tr = F1_score(X_tr,theta,y_tr);
%Validation score
score_before_val = F1_score(X_val,initial_theta,y_val);
score_after_val = F1_score(X_val,theta,y_val);

%Plotting the decision-boundary
%plotDecisionBoundary(theta,X_val,y_val)
%gplotmatrix(features(:,5),features(:,1),activity);

%2.3 
%adding polonial features
X_pol = mapFeature(X1_tr,X2_tr,6);
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
score_vector_val(i) = F1_score(X_val,theta,y_val);
end

%Plotting lambda and F1-score for training and validation
%figure
%hold on
%plot(lambda_pol,score_vector_val)
%plot(lambda_pol,score_vector_tr)
%axis([0 700 0 1])
%legend('Cross validation','Training')
%When lambda increases --> High bias

[val, idx] = max(score_vector_val);
plotDecisionBoundary(theta_vector(idx,:),X_val,y_val)

% going from 8 features to quad ratic representation
% anv�nd x2fx



