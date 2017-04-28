function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

C_candicate = [0.01 0.03 0.1 0.3 1.0 3.0 10 30];
sigma_candicate = [0.01 0.03 0.1 0.3 1.0 3.0 10 30];


predict_errors = zeros(length(C_candicate), length(sigma_candicate));

for c_i=1:length(C_candicate)
    for s_j = 1:length(sigma_candicate)
       C = C_candicate(c_i);
       sigma = sigma_candicate(s_j);
      kerfun = @(x1, x2) gaussianKernel(x1, x2, sigma);
      model = svmTrain(X, y, C, kerfun);
      y_predict = svmPredict(model, Xval);
      
      predict_errors(c_i, s_j) = mean(double(y_predict ~= yval));
    end
end

a = predict_errors

[min_vals, min_lines] = min(predict_errors, [] ,1);
[minv, min_col] = min(min_vals);

min_line = min_lines(min_col);

C = C_candicate(min_line)
sigma = sigma_candicate(min_col)

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
