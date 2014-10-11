% this code adapted from:
% O. Ludwig, D. Delgado, V. Goncalves, and U. Nunes, "Trainable 
% Classifier-Fusion Schemes: An Application To Pedestrian Detection,"
% In: 12th International IEEE Conference On Intelligent Transportation 
% Systems, 2009, St. Louis, 2009. V. 1. P. 432-437.

function H = extracthog(I)
Ig = applygaussian(I);
window = 3; % set the window size in the x and y directions
% the number of histogram bins to use - this number recommended by dalal
% and triggs for human detection. since our experiments are similar, we
% also employ nine bins
nbins = 9;
[rows,cols] = size(Ig);
H = zeros(window^2 * nbins,1); % initialize the returned hog descriptor
sx = floor(cols / (window + 1)); % step sizes for indexing scheme
sy = floor(rows / (window + 1));
c = 0;
[a,m] = gradientfilterimage(Ig);
for i = 0:window - 1 % complete pass through the image using indexing scheme
    for j = 0:window - 1
        c = c + 1;
        angle = a(i * sy + 1:(i + 2) * sy,j * sx + 1:(j + 2) * sx);
        magnitude = m(i * sy + 1:(i + 2) * sy,j * sx + 1:(j + 2) * sx);
        H((c - 1) * nbins + 1:c * nbins,1) = binangles(angle,magnitude,nbins);
    end
end
end

function Img = applygaussian(Im)
Im = double(Im); % image cast to double
gauss = fspecial('gaussian',[5 5],2); % apply gaussian filtering to image
Img = imfilter(Im,gauss,'same');
end

function h = binangles(angle,magnitude,nbins)
angle = angle(:);
magnitude = magnitude(:);
b = 0; % this bin identifier
h = zeros(nbins,1);
% bin angles in intervals of twenty degrees
for anglelimit = -pi + 2*pi/nbins:2*pi/nbins:pi
    b = b + 1;
    index = find(angle <= anglelimit);
    angle(index) = 100;
    h(b) = h(b) + sum(magnitude(index));
end
h = h / (norm(h) + eps);
end

function [a,m] = gradientfilterimage(Img)
fx = [-1,0,1]; % gradient filters recommended by dalal and triggs
fy = -fx';
gx = imfilter(double(Img),fx);
gy = imfilter(double(Img),fy);
a = atan2(gy,gx);
m = sqrt(gy.^2 + gx.^2);
end