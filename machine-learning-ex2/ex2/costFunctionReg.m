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

%h = sigmoid(X*theta);

%k = ones(1,size(theta));
%J = 1/m * (-y'*log(h) - (1-y)'*log(1-h)) + (lambda/(2*m))*(k*theta.^2);

%X_0 = X(:,1);
%theta_0 = theta(1,:);
%h_0 = sigmoid(X_0*theta_0);
%grad_0 = 1/m * X_0'*(h_0-y);

%X_=X(:,2:columns(X));
%theta_ = [0; theta(2:length(theta),:)];
%h_ = sigmoid(X_*theta_);
%grad = 1/m * X_'*(h_-y) + (lambda/m)*theta_;
%grad = 1/m * X'*(h-y) + (lambda/m)*theta_;

%grad = [grad_0; grad];


h = sigmoid(X*theta);
theta_ = [0; theta(2:length(theta),:)];
J = 1/m * (-y'*log(h) - (1-y)'*log(1-h)) + (lambda/(2*m))*(theta_'*theta_);
grad = 1/m * X'*(h-y) + (lambda/m)*theta_;

% =============================================================

end
