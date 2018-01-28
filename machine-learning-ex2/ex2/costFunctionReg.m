function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta);

% Since regularization should not take into account the first element
% of theta, i.e. theta(0), we create a copy of the theta vector with
% the first row zeroed out. We plug this new vector into the original
% cost and gradient equations with the addition of the regularization
% term. With the first element of theta zeroed out, we effectively
% cancel out the regularization for the first element of theta.
theta_ = [0; theta(2:length(theta),:)];

% the term theta_'*theta is a vectorization of the summation of theta_^squared
J = 1/m * (-y'*log(h) - (1-y)'*log(1-h)) + (lambda/(2*m))*(theta_'*theta_);
grad = 1/m * X'*(h-y) + (lambda/m)*theta_;

% =============================================================

end
