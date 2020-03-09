% Summary: this function performs a backward propagation operation to
% update the ANN model using the gradient of the cost function, learning
% rate, and L2 regularization constant
%
% Inputs: ANN model struct, cost gradient w.r.t. activation, the L2
% regularization constant, and learning rate
%
% Outputs: ANN model with updated weight and bias values

function ANN = RunBackward(ANN, dc_da, alpha, L2)

% References:
% https://brilliant.org/wiki/backpropagation/
% https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/
% https://towardsdatascience.com/understanding-the-scaling-of-l%C2%B2-regularization-in-the-context-of-neural-networks-e3d25f8b50db

% Calculate derivative of activations w.r.t. weights for the hidden and
% output layers
da2_dw = ANN.layer2_a.*(1-ANN.layer2_a);
da3_dw = ANN.layer3_a.*(1-ANN.layer3_a);

% Calculate hidden layer error and update bias and weights parameters
ANN.layer2_delta = (ANN.layer3_weights)'*(dc_da.*da3_dw).*da2_dw;
ANN.layer2_bias = ANN.layer2_bias - alpha*ANN.layer2_delta;
ANN.layer2_weights = (1 - L2)*ANN.layer2_weights - alpha*ANN.layer2_delta*(ANN.layer1_a)';

% Calculate output layer error and update bias and weights parameters
ANN.layer3_delta = dc_da.*da3_dw;
ANN.layer3_bias = ANN.layer3_bias - alpha*ANN.layer3_delta;
ANN.layer3_weights = (1-L2)*ANN.layer3_weights - alpha*ANN.layer3_delta*(ANN.layer2_a)';

end

