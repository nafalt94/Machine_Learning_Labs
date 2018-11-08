function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

[rows, columns] = size(X);
mu=zeros(columns,1);
sigma=zeros(columns,1);
X_norm=X;

for i =1:columns
   mu(i) = mean(X(:,i));
    X_norm(:,i) = X_norm(:,i) - mu(i); %bsxfun(@minus, X(:,i), mu(i));
    
     sigma(i) = std(X_norm(:,i));
 X_norm(:,i) = X_norm(:,i)/sigma(i); %bsxfun(@rdivide, X_norm(i,:), sigma(i));
 
end

 mu = mean(X);
% X_norm = bsxfun(@minus, X, mu);
% 
 sigma = std(X_norm);
% X_norm = bsxfun(@rdivide, X_norm, sigma);


% ============================================================

end
