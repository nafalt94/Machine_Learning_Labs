load('C:\Users\Gustav\Desktop\Plugg Leuven\Machine Learning\Lab1\Master-LR\Dataset\Features.mat')
load('C:\Users\Gustav\Desktop\Plugg Leuven\Machine Learning\Lab1\Master-LR\Dataset\Label.mat')
load('C:\Users\Gustav\Desktop\Plugg Leuven\Machine Learning\Lab1\Master-LR\Dataset\subject.mat')

%Excercise 1
feat_tr = features(1:round((length(label)*0.4)),:); 
feat_val = features(round((length(label)*0.4))+1:round((length(label)*0.7)),:); 
feat_test = features(round((length(label)*0.7))+1:length(label),:); 

label_tr = label(1:round((length(label)*0.4)));
label_val= label(round((length(label)*0.4))+1:round((length(label)*0.7))); 
label_test = label(round((length(label)*0.7))+1:length(label)); 

% 1 vs all classification
activity = label == 6;
activity = double(activity);
plotlabel = label;


%figure;
%gplotmatrix(features(:,6),features(:,2),activity);
%Feature nr5
X1 = feat_tr(:,6);
X2=feat_tr(:,2);
X = ones(length(label_tr),3);
X(:,2) = X1;
X(:,3) = X2;

[m, n] = size(X);

y = activity(1:round((length(label)*0.4)),:); 

lambda = 0;
%initial_theta = zeros(n + 1, 1);
initial_theta = zeros(size(X, 2), 1);

%[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
options = optimset('GradObj','on','MaxIter',400);
[theta, cost, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)),initial_theta,options);

%theta = fminunc(costFunctionReg(initial_theta, X, y, lambda),X0);    


z1 = X*(initial_theta);
score_before = F1_score(z1,y);

z2 = X*(theta);
score_after = F1_score(z2,y);

%plotDecisionBoundary(theta,X,y)
%gplotmatrix(features(:,5),features(:,1),activity);

%2.3 
%adding polonial features
X_pol = mapFeature(X1,X2,6);
%lambda_pol = (3^(-10)):.2:(3^(10));

lambda_pol(1) = 3^(-10);
i = 1;
while(lambda_pol(i)<(3^(10)))
   i=i+1;
    lambda_pol(i) = lambda_pol(i-1)*2; 
end




