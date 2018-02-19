function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X * theta;
errors = h - y;
J = 1/(2*m)*(errors)'*(errors);
grad = (1/m) * X'*errors;

% Add regularization
% Since regularization should not take into account the first element
% of theta, i.e. theta(0), we create a copy of the theta vector with
% the first row zeroed out. We compute the regularization portion using
% this version of theta with the first row zeroed out.
theta_ = [0; theta(2:length(theta),:)];
J = J + (lambda/(2*m))*(theta_'*theta_);
grad = grad + (lambda/m)*theta_;










% =========================================================================

grad = grad(:);

end
