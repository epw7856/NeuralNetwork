% Summary: this function tests the accuracy of the ANN with the test image
% set using forward propagation. The largest output node activation for each
% image is compared against its actual label for verification.
%
% Inputs: ANN model struct, full image matrix, labels for the image matrix,
% and the testing image indices
%
% Outputs: accuracy percentage of the ANN when testing all images

function accuracy = RunTest(ANN, image_matrix, labels, testing_data)

% Execute forward propagation with the test image subset
ANN = RunForward(ANN, image_matrix(:, testing_data));

% The predicted classification is a 1x1x310 matrix with each element in the
% 3rd dimension corresponding to the index of the output node with the 
% largest activation. The actual classification is determined using the
% same method because the index of the appropriate digit/letter will be
% returned where the 1 is located in the labels matrix for the testing 
% images.
[~, predicted] = maxk(ANN.layer3_a, 1, 1);
[~, actual] = maxk(labels(:, 1, testing_data), 1, 1);

% Transform the predicted and actual labels into a row vector for easy
% comparison
predicted = permute(predicted, [3,2,1]);
actual = permute(actual, [3,2,1]);

% Compare the predicted and actual labels and calculate classification 
% accuracy
correct_count = 0;
for i=1:length(predicted)
    if predicted(i) == actual(i)
        correct_count = correct_count + 1;
    end    
end

accuracy = (correct_count / length(predicted))*100;

end

