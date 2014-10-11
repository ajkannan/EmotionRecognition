% This function extracts and saves crops of the eyes and mouth for each
% image in the dataset.  This function was designed and implemented
% completely by our team. 

function extracteyesmouth
close all;
% Run with 9 eye templates (at least one for each emotion) and 15 mouth
% templates, 2 for each emotion, 3 for joy, since joy has a lot of
% variation in mouth appearance.
num_eye_templates = 9;
num_mouth_templates = 15;

% Size of cropped mouth templates (after resizing)
mouth_height = 100;
mouth_length = 200;
scale = [mouth_height, mouth_length];

% Approximate size of cropped eye templates
eye_height = 100;
eye_length = 300;
offset = 15; % To provide a border around the cross correlated crop chosen

% Create the Gaussian filter for the templates and images.
hsize = 10;
sigma = 5;
h = fspecial('gaussian', hsize, sigma);
eye_template = cell(num_eye_templates,1);
mouth_template = cell(num_mouth_templates);

% Templates gathered from raw image data, but the cropped images are 
% larger.  Hence we scale by 1.5
scale_factor = 1.5;

% Mouth templates
mouth_template{1} = imresize(imread('./Templates/disgust.jpg'),scale);
mouth_template{2} = imresize(imread('./Templates/disgust2.jpg'),scale);
mouth_template{3} = imresize(imread('./Templates/fear.jpg'),scale);
mouth_template{4} = imresize(imread('./Templates/fear2.jpg'),scale);
mouth_template{5} = imresize(imread('./Templates/happy.jpg'),scale);
mouth_template{6} = imresize(imread('./Templates/happy2.jpg'),scale);
mouth_template{7} = imresize(imread('./Templates/happy3.jpg'),scale);
mouth_template{8} = imresize(imread('./Templates/angry.jpg'),scale);
mouth_template{9} = imresize(imread('./Templates/angry2.jpg'),scale);
mouth_template{10} = imresize(imread('./Templates/neutral.jpg'),scale);
mouth_template{11} = imresize(imread('./Templates/neutral2.jpg'),scale);
mouth_template{12} = imresize(imread('./Templates/sad.jpg'),scale);
mouth_template{13} = imresize(imread('./Templates/sad2.jpg'),scale);
mouth_template{14} = imresize(imread('./Templates/surprise.jpg'),scale);
mouth_template{15} = imresize(imread('./Templates/surprise2.jpg'),scale);

% Eye templates
eye_template{1} = imresize(imread('./Templates/eye1.jpg'),scale_factor);
eye_template{2} = imresize(imread('./Templates/eye2.jpg'),scale_factor);
eye_template{3} = imresize(imread('./Templates/eye3.jpg'),scale_factor);
eye_template{4} = imresize(imread('./Templates/eye4.jpg'),scale_factor);
eye_template{5} = imresize(imread('./Templates/eye5.jpg'),scale_factor);
eye_template{6} = imresize(imread('./Templates/eye6.jpg'),scale_factor);
eye_template{7} = imresize(imread('./Templates/eye7.jpg'),scale_factor);
eye_template{8} = imresize(imread('./Templates/eye8.jpg'),scale_factor);
eye_template{9} = imresize(imread('./Templates/eye9.jpg'),scale_factor);

% Blur the templates.  We found slightly better correlation results after
% applying a Gaussian filter, removing noise that could cause the matching
% to misfire
for i = 1:num_eye_templates
    eye_template{i} = imfilter(eye_template{i},h);
end
for i = 1:num_mouth_templates
    mouth_template{i} = imfilter(mouth_template{i},h);
end

% Load the cropped images
load('imagedata_crop.mat');
cropped_eyes = cell(size(train_x,1) + size(test_x,1), 1);
cropped_mouth = cell(size(train_x,1) + size(test_x,1), 1);

% Loop through all the images to extract the eyes and mouth
for image = 1:length(cropped_eyes)
    % Figure out whether to take image from train or test data
    if image > size(train_x,1)
        index = image - size(train_x,1);
        I = reshape(test_x(index,:), 300, 348) / 255;
    else
        I = reshape(train_x(image,:), 300, 348) / 255;
    end
    
    % Some of the images in the dataset are rgb, not grayscale, so we
    % convert to rgb.
    if(size(I,3) == 3)
        I = rgb2gray(I);
    end
    
    % Filter the image
    I_filtered = imfilter(I, h);
    
    % Get the coordinates pertinant to finding the limits of the eye
    [eye_left, eye_bottom] = find_eye_bounding_box(num_eye_templates, eye_template, I_filtered, eye_height);
    x_start = eye_left;
    x_end = eye_left + eye_length / 2;
    y_start = eye_bottom - eye_height - 2*offset;
    y_end = eye_bottom;
    
    % Ensure that the points are within the limits of the image
    [x_start,x_end, y_start, y_end] = point_check(x_start, x_end, y_start, y_end);
    cropped_eyes{image} = I(y_start:y_end,x_start:x_end);
    
    %figure;imshow(cropped_eyes{image});
    %pause(.3)
    
    % Set bounds for mouth finding below
    eye_left = x_start;
    eye_right = x_end + eye_length / 2;
    if eye_right > 348
        eye_right = 348;
    end
    eye_bottom = y_end;
    
    % Run cross correlation on mouths using the templates
    mouth_results = zeros(num_mouth_templates, 2);
    for j = 1:size(mouth_template,1)
        [max_x, max_y] = mouth_corr(mouth_template{j},I_filtered(eye_bottom:end,eye_left:eye_right));
        mouth_results(j,1) = max_x;
        mouth_results(j,2) = max_y;
    end    
    
    % Choose the best result through filtering out outliers
    [x,y] = choose_position(mouth_results, eye_bottom);

    % Calculate bounding box for mouth
    x_start = x + eye_left;
    x_end = x_start + mouth_length;
    y_start = y + eye_bottom;
    y_end = y_start + mouth_height;
    
    % Check that the bounding box lies entirely in the image
    [x_start, x_end, y_start, y_end] = point_check(x_start, x_end, y_start, y_end);
    
    % Crop the mouth and store it
    try
        cropped_mouth{image} = I(y_start:y_end, x_start:x_end);
    catch e
        fprintf('Image %d failed\n',image);
    end
    %figure;imshow(cropped_mouth{image});
    %pause(.3);
end

save('cropped_eyes.mat','cropped_eyes');
save('cropped_mouth.mat','cropped_mouth');
end


% Run the cross correlation on the mouth
function [max_x, max_y] = mouth_corr(template, I)
try 
    % If the template is larger than the restricted image area, the eye
    % finding did not perform well on the specific image.  However, we do
    % not want to error and stop, so we throw out the image mouth by
    % setting it to NaN.
    C = normxcorr2(template,I);
    [mt, nt] = size(template);
    Ctrimmed = C(mt - 1:end,nt - 1:end);
    [~,cindex] = max(Ctrimmed(:));
    [max_x,max_y] = ind2sub(size(Ctrimmed),cindex);
catch e
    max_x = NaN;
    max_y = NaN;
end
end


% Choose the best cross-correlation results based on a voting-like procedure,
% removing outliers and then taking the median.
function [x,y] = choose_position(mouth_results, eye_bottom)
for i = 1:size(mouth_results,1)
    if mouth_results(i,1) > 200
        mouth_results(i,1) = NaN;
        mouth_results(i,2) = NaN;
    end
end

% Find the median values 
x = floor(nanmedian(mouth_results(:,1)));
[i,~] = find(mouth_results(:,1) == x); % Find the corresponding y-coordinate
y = floor(median(mouth_results(i,2)));
end


% Find the coordinates of eye bounding box
function [eye_left, eye_bottom] = find_eye_bounding_box(num_eye_templates, eye_template, I_filtered, eye_height)
eye_results = zeros(num_eye_templates,3);

% Run the 9 templates on the image
for j = 1:num_eye_templates
    [max_x, max_y, max_corr] = find_eyes(eye_template{j},I_filtered);
    eye_results(j,1) = max_x;
    eye_results(j,2) = max_y;
    eye_results(j,3) = max_corr;
end 

% Take the median of the x-coords to exclude influence of outliers
eye_left = median(eye_results(:,1));
[i,~] = find(eye_results(:,1) == eye_left); % Find the corresponding y-coordinate
eye_bottom = floor(median(eye_results(i,2))) + eye_height;
end


% Run the cross-correlation between the template and the image
function [max_x, max_y, max_corr] = find_eyes(template,I)
C = normxcorr2(template,I);
[mt, nt] = size(template);
Ctrimmed = C(mt - 1:end,nt - 1:end);
[max_corr,cindex] = max(Ctrimmed(:));
[max_x,max_y] = ind2sub(size(Ctrimmed),cindex);
end


% Ensures that the image bounds do not exceed the dimensions of the image
function [x_start, x_end, y_start, y_end] = point_check(x_start,x_end,y_start,y_end)
% All images from imagedata_crop are 300 by 348 pixels
x_lim = 348;
y_lim = 300;

% Adjust the start and end of the range to choose for the cropped eye if
% needed
if x_start < 1
    x_start = 1;
end
if x_end > x_lim
    x_end = x_lim;
end
if y_start < 1
    y_start = 1;
end
if y_end > y_lim;
    y_end = y_lim;
end
end
