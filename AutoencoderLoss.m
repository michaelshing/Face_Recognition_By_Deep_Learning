function [loss,grad] = AutoencoderLoss(theta, hiddenSize, visibleSize,lambda,...
                                     targetActivation, beta, noisedata,data,vsigmoid,hsigmoid)

% The input theta is a vector because minFunc only deal with vectors. In
% this step, we will convert theta to matrix format such that they follow
% the notation in the lecture notes.
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Loss and gradient variables (your code needs to compute these values)
m = size(data, 2);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the loss for the Sparse Autoencoder and gradients
%                W1grad, W2grad, b1grad, b2grad
%
%  Hint: 1) data(:,i) is the i-th example
%        2) your computation of loss and gradients should match the size
%        above for loss, W1grad, W2grad, b1grad, b2grad

% z2 = W1 * x + b1
% a2 = f(z2)
% z3 = W2 * a2 + b2
% h_Wb = a3 = f(z3)

z2 = W1 * noisedata + repmat(b1, [1, m]);

if hsigmoid
    a2 = sigmoid(z2);
else
    a2=z2;
end

z3 = W2 * a2 + repmat(b2, [1, m]);

if vsigmoid
    a3 = sigmoid(z3);
else
    a3=z3;
end

rhohats = mean(a2,2);
%rho = targetActivation;
%KLsum = sum(rho * log(rho ./ rhohats) + (1-rho) * log((1-rho) ./ (1-rhohats)));
rho = targetActivation;
KLsum=0;
squares = (a3 - data).^2;
squared_err_J = (1/2) * (1/m) * sum(squares(:));
weight_decay_J = (lambda/2) * (sum(W1(:).^2) + sum(W2(:).^2));
sparsity_J = beta * KLsum;

loss = squared_err_J + weight_decay_J + sparsity_J;

if nargout > 1 
    % delta3 = -(data - a3) .* fprime(z3);
    % but fprime(z3) = a3 * (1-a3)
    delta3 = -(data - a3);
    if vsigmoid 
        delta3=delta3.* a3 .* (1-a3);
    end
    beta_term = beta * (- rho ./ rhohats + (1-rho) ./ (1-rhohats));
    
    delta2 = ((W2' * delta3) + repmat(beta_term, [1,m]) );
    if hsigmoid
        delta2=delta2.* a2 .* (1-a2);
    end
    
    W2grad = (1/m) * delta3 * a2' + lambda * W2;
    b2grad = (1/m) * sum(delta3, 2);
    W1grad = (1/m) * delta2 * noisedata' + lambda * W1;
    b1grad = (1/m) * sum(delta2, 2);

    %-------------------------------------------------------------------
    % Convert weights and bias gradients to a compressed form
    % This step will concatenate and flatten all your gradients to a vector
    % which can be used in the optimization method.
    grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
end

end
%-------------------------------------------------------------------
% We are giving you the sigmoid function, you may find this function
% useful in your computation of the loss and the gradients.
function sigm = sigmoid(x)

    sigm = 1 ./ (1 + exp(-x));
end
