function [J, grad] = costFunctionReg(theta, X, y, lambda)
m = length(y);
% Initialize some useful values
l = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE ======================
% You need to finish the cost function and calculate the gradient result in
% here.

z = X*(theta);
h = sigmoid(z);

A = (-y'*(log(h))-(1-y)'*(log(1-h)));

thetabajs = theta;
thetabajs(1) = 0;

%fattar inte riktigt varför ^2 funkar...
J = ((A))/m + ((lambda/(2*m))*(thetabajs'*thetabajs));

B = (h - y)';

grad = ((1/m)*B*X) + lambda*thetabajs'/m; 
% =============================================================

end
