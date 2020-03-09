%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Artificial Neural Network for Facial Recognition
% Author: Eric Walker
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc

% Set static variables and parameters
sample_sets = 62;                               % Num of image directories
images_per_set = 55;                            % Images per directory
total_images = images_per_set*sample_sets;      % Total images
image_size = 36*36;                             % Processed image size
hidden_layer = 100;                             % Nodes in hidden layer
alpha = .032;                                   % Learning Rate
L2 = 0.00001;                                   % L2 Regularization Weight
cost_convergence = 0.006;                       % Criterion to end training/testing. Adjust this to end training sooner.
test_interval = 1000;                           % Periodicity to test accuracy of ANN using test images


% Phase 1: Load and Process Images
% If the image matrix is not already loaded, load and process the database
% images. Processing the images from the database requires several minutes
% but a progress bar updates status periodically.
if (exist('image_matrix', 'var') == 0)
    image_matrix = ProcessImages(total_images, image_size);
end
% End Phase 1


% Phase 2: Set Training and Testing Datasets and Classification Labels
% Select 50 random images from each of the 62 image sets. The total number
% of training images will be 3100. This algorithm picks 50 random, sorted
% numbers between 1 and 55 for each set. This ensures that 50 random images
% from each set are selected. The values inside the training_data variable
% are simply indices that will be used to extract feature vectors from 
% image_matrix.
for i = 1:sample_sets
    training_data((i-1)*50+1:(i-1)*50+50) = sort(randperm(55, 50)) + 55*(i-1);
end

% The images that weren't chosen for testing are selected for the testing
% set. There is a comparison made between a simple vector that goes from 1
% to 3410 to see which indices are not in training_data.
testing_data = setdiff(([1:total_images]), training_data);

% A matrix must be present that contains information on the actual labels
% of the training and testing data. Neural networks function via tensors,
% or 3d matrices, so the label information must be constructed in 3d. The
% label matrix is a 62x1x3410 matrix (a 62x1 vector for each image). A 1 is
% populated in the appropriate row for the label of the image with the rest 
% of the rows being equal to 0.
labels = [zeros(62, 1, total_images)];
for i = 1:total_images
    index = ceil(i/55);
    labels(index,1,i) = 1;
end
% End Phase 2


% Phase 3: Initialize ANN Parameters
% The neural network is actually a struct containing information for all
% layers such as weights, biases, outputs, etc. The input layer has a
% number of nodes equal to the image_size variable, the hidden layer has
% 100 nodes, and the output layer has a number of nodes equal to the
% sample_sets variable (62). Weights are randomly initialized
% between -1 and 1. Biases are initialized to 1. Delta parameters for back
% propagation are initialized to 0.
clear ANN
ANN = [];

% Setup ANN hidden layer parameters (Layer 2)
a = -1; b = 1;
ANN.layer2_weights = (b-a)*rand(hidden_layer, image_size) - b;
ANN.layer2_bias = ones(hidden_layer, 1);
ANN.layer2_delta = zeros(hidden_layer, 1);

% Setup ANN output layer parameters (Layer 3)
a = -1*sqrt(6)/sqrt(162); b = 1*sqrt(6)/sqrt(162);
ANN.layer3_weights = (b-a)*rand(sample_sets, hidden_layer) - b;
ANN.layer3_bias = ones(sample_sets, 1);
ANN.layer3_delta = zeros(sample_sets, 1);
% End Phase 3


% Start the alogrithm to train and test the ANN.
% Set initial cost value to 1.
clear cost epoch_count accuracy
cost = 1; epoch_count = 1; accuracy = 0; epoch_testing = 0;

% Weighted Cost is used as the convergence criteria to stop
% training/testing. The program will exit this loop and terminate training
% only when this criteria is met.
while (cost(end) > cost_convergence)  
    
    cost_last_iteration = cost(epoch_count);
    epoch_count = epoch_count + 1;
    
    
    % Phase 4: Run Forward Propagation with a Training Image
    % Use "online-mode" to update model (i.e. provide samples 1 by 1 into
    % the ANN and update the model accordingly each time). Randomly choose
    % the training image index.
    chosen_training_image = randi(size(training_data, 2));
    
    % Run forward propagation with the current ANN model and the selected
    % training image as the feature vector. Rotate the chosen image by a
    % randomly selected degree using imrotate function. This provides
    % higher fideliy training to the model for input variations.
    rotated_image_vector = reshape(imrotate((reshape(image_matrix(:, training_data(chosen_training_image)),...
                        sqrt(image_size), sqrt(image_size))), randi(36) - 18, 'bilinear', 'crop'), image_size, 1);
    ANN = RunForward(ANN, rotated_image_vector);
    % End Phase 4
    
    
    % Phase 5: Calculate Cost and Cost Gradient
    % Obtain the label vector for the training image and calculate cost 
    % value for the training example using cross entropy.
    % Reference: http://neuralnetworksanddeeplearning.com/chap3.html
    y = labels(:, training_data(chosen_training_image));
    cost(epoch_count) = -1*mean((y.*log(ANN.layer3_a) + (1-y).*log(1-ANN.layer3_a)), 1);
    
    % Filter the new cross entropy value to provide a more accurate
    % representation of the actual cost of the model.
    % Reference: https://www.mathworks.com/help/dsp/ug/sliding-window-method-and-exponential-weighting-method.html
    cost(epoch_count) = .01*cost(epoch_count) + .99*cost_last_iteration;
    
    % Calculate the gradient of the cost function w.r.t. the nodal activation
    % Reference: https://medium.com/@pdquant/all-the-backpropagation-derivatives-d5275f727f60
    dc_da = (ANN.layer3_a - y)./(ANN.layer3_a.*(1-ANN.layer3_a));
    % End Phase 5
    
    
    % Phase 6: Run Back Propagation
    ANN = RunBackward(ANN, dc_da, alpha, L2);
    % End Phase 6
    
    
    % Phase 7: Periodically test the accuracy of the model
    % Testing the ANN with the 310 test images after every epoch is
    % expensive so it will be performed after every 1000 epochs. 
    if (mod(epoch_count, test_interval) == 0)
        
        % Store the accuracy for this test by appending it to the
        % previously calculated accuracies. Also maintain the epoch count
        % for when this test was run
        accuracy = [accuracy RunTest(ANN, image_matrix, labels, testing_data)];
        epoch_testing = [epoch_testing epoch_count];
        
        % Print results of the testing
        fprintf('Epoch Count: %d\n', epoch_count);
        fprintf('Cost: %.6f\n', cost(epoch_count));
        fprintf('Accuracy: %.2f %%\n\n\n', accuracy(end));
        
    end
    % End Phase 7

    
end


% The cost function convergence criteria has been met and the training of
% the ANN has completed. Run a classification of the test images to
% calculate the final accuracy.
accuracy = [accuracy RunTest(ANN, image_matrix, labels, testing_data)];
epoch_testing = [epoch_testing epoch_count];

% Print final ANN performance metrics
fprintf('Convergence criteria satisfied. Training complete.\n');
fprintf('Final Epoch Count: %d\n', epoch_count);
fprintf('Final Cost: %.6f\n', cost(epoch_count));
fprintf('Final Accuracy: %.2f %%\n\n\n', accuracy(end));

% Plot the Cost Function Output over the epochs
figure(1)
plot(cost)
title("Cost Estimation Error Over Training Period")
xlabel("Epoch Number")
ylabel("Cross Entropy")
xlim([0 epoch_count])

% Plot the accuracy of the ANN over the epochs
figure(2)
plot(epoch_testing, accuracy)
title("Classification Accuracy")
xlabel("Epoch Number")
ylabel("Accuracy Percentage")
xlim([0 epoch_count])
ylim([0 100])

