% Summary: this function loads all images from the database and performs
% multiple processing algorithms on them. The images are reshaped into 
% column vectors and concatenated with each other to form an NxM matrix, 
% where N is the final image dimensions multiplied together and M is the 
% total number of images.
%
% Inputs: total number of images in the database, and the final processed
% image size
%
% Outputs: matrix containing concatenated column vectors of images

function image_matrix = ProcessImages(total_images, downsampled_image_size)

% The image matrix takes the shape of (N*N)x(total_images), where N*N is
% the size of a single downsampled, processed image.
image_matrix = zeros(downsampled_image_size, total_images);

% Recursively obtain all contents from within the image database path
[~, allContents] = fileattrib('EnglishHnd\*');  

for i=1:length(allContents)
    
    % Only keep the paths that are files, not folders. Set empty cells for
    % non-files.
    if isfile(allContents(1, i).Name)
        imagePaths{i, 1} = allContents(1, i).Name;
    else
        imagePaths{i, 1} = '';
    end
    
end

% Remove the empty cells in the imagePaths array
imagePaths(cellfun('isempty', imagePaths)) = [];

clear allContents

% Create a progress bar to update the status on loading/processing images
progress = waitbar(0, 'Processing Images');

% Process each image within the path list
for i = 1:total_images
    
    waitbar(i/total_images, progress, 'Processing Images');
    
    % Step 1: Load images, convert to grayscale, and then invert image
    % colors (i.e. change white backgrounds to black and black lettering 
    % to white)
    image = rgb2gray(imread(imagePaths{i}));
    image = double(imcomplement(image));
    image_size = size(image);
    
    % Step 2: Remove empty rows and cols (if a row or col of pixels is all 0,
    % remove it). The result is a cropped image of the handwritten digit or
    % letter.
    image = image(sum(image, 2) > 0, sum(image, 1) > 0);
    
    % Step 3: Scale the image up and pad the images with 0's in both 
    % dimenions to achieve 1200x1200 image size.
    resize_dimension =  min(image_size./size(image));
    image = imresize(image, resize_dimension);
    image = padarray(image, [1200-(size(image, 1)) 1200-(size(image, 2))], 0, 'post');

    % Step 4: The processed image is now 1200x1200 pixels but the 
    % digit/letter is offset in the upper left corner. The digit/letter 
    % must be centered in the middle of the image. References for this
    % centering technique was discovered via Mathworks forums.
    % References:
    % https://www.mathworks.com/matlabcentral/answers/27370-moving-the-center-of-an-image-from-point-to-another
    % https://www.mathworks.com/matlabcentral/answers/281198-problem-in-alignment-of-images-and-taking-average
    image_size = size(image);
    [cols, rows] = meshgrid([1:image_size(2)], [1:image_size(1)]);
    image_sum = sum(sum(image));
    horizontal_shift = sum(sum(image.*cols)) / image_sum;
    vertical_shift = sum(sum(image.*rows)) / image_sum;
    x_translation = image_size(2)/2 - floor(horizontal_shift);
    y_translation = image_size(1)/2 - floor(vertical_shift);
    image = imtranslate(image, [x_translation, y_translation]);
    
    % Step 5: Downsample the centered, high-resolution image to a much
    % smaller dimension and then convert to a column vector. The purpose
    % of downsampling is to reduce the number of input layer nodes needed
    % in the neural network. Downsampling occurs in both dimensions
    % according to a factor that achieves the final image dimensions from
    % 1200x1200 pixels.
    downsampled_image = downsample(downsample(image(:,:,1), 34)', 34)';
    image_matrix(:,i) = reshape(downsampled_image, [length(image_matrix(:,1)), 1]);
    
end

% Normalize the image pixel values between 0 and 1.
image_matrix = double(image_matrix / 255.0);

% Close the progress bar.
close(progress)
clear image downsampled_image imagePaths

end