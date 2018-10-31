function [theta, cost, exit_flag] = training(X_tr, y_tr, lambda)

%[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

initial_theta = zeros(size(X_tr, 2), 1); 

options = optimset('GradObj','on','MaxIter',400);
[theta, cost, exit_flag] = fminunc(@(t)(costFunctionReg(t, X_tr, y_tr, lambda)),initial_theta,options);
end

