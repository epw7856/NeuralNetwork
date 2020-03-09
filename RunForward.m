% Summary: this function performs a forward propagation operation of image
% vector(s) through the ANN.
%
% Inputs: ANN model struct and the image/feacture vectors in  MxN matrix 
% format (one or many images can be processed at once)
%
% Outputs: ANN model with updated node activation values

function ANN = RunForward(ANN, images)

% Set input layer node values (i.e. image values): 
% ANN's work with 3d matrices so the input images, whether it is a single 
% Mx1 column vector or an MxN set of images, must be transformed into a 3d
% matrix with all image vectors stacked in the 3rd dimension.
ANN.layer1_a = permute(images, [1,3,2]);

% Calculate hidden layer node activation:
% z = w*x+b  -> transpose the x-y dimensions of the inputs, element-wise
% multiply with the hidden layer weights, sum across rows, and add bias
% term to each row. The bias term has to be expanded in 3 dimensions.
% a = sigma(z) = 1 / (1 + e^(-z))
ANN.layer2_z = sum(ANN.layer2_weights.*permute(ANN.layer1_a, [2,1,3]), 2);
ANN.layer2_z = ANN.layer2_z + repmat(ANN.layer2_bias, 1, 1, size(images, 2));
ANN.layer2_a = 1./(1 + exp(-ANN.layer2_z));

% Calculate output layer node activation:
% Same equations as hidden layer. These activations correspond to the
% output of the ANN.
ANN.layer3_z = sum(ANN.layer3_weights.*permute(ANN.layer2_a, [2,1,3]), 2);
ANN.layer3_z = ANN.layer3_z + repmat(ANN.layer3_bias, 1, 1, size(images, 2));
ANN.layer3_a = 1./(1 + exp(-ANN.layer3_z));

end