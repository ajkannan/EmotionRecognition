function loadpartitionedsiftdata()
% load the labels from the text file
y = load('Labels/emotion_labels.txt'); 

% get the filenames from the text file
c = textscan(fopen('Labels/associated_pic_filenames.txt'),'%s');

% the number of files we have
N = length(c{1});

% the indices of those images which did not give an error
index = zeros(N,1);

% the indices of the sift features belonging to a particular image

indices = cell(length(y),1);

% big matrix of sift descriptors - 128 dimensions each
label = [];

% the ranges for the dictionary size to use in sift
k = 20:10:60;

% for each image
for i = 1:N
    try
        fprintf('image id: %d\n',i);
	% detect mouth and eyes using the viola-jones algorithm
        detector = buildDetector();
        I = imread(strcat(['../.' char(c{1}(i))]));
        bbox = detectFaceParts(detector,I);
	
	% mouth and eyes respectively
        Im = convertg(imcrop(I,bbox(end,13:16)));
        Ie = convertg(imcrop(I,bbox(end,5:8)));
        
	% get sift descriptors for both the extracted eyes and mouth
        [~,dm,~] = sift2(Im);
        [~,de,~] = sift2(Ie);

	% concatenate features
        descriptor = [dm;de];
	
	% keep track of the indices into label for this image
        indices{i} = size(label,1) + (1:size(descriptor,1));
        label = [label;descriptor];
	
	% the image finished without error
        index(i) = 1;
    catch e
        disp(e)
    end
end

% for each dictionary size
for ki = 1:length(k)
    fprintf('clustering with k = %d\n',k(ki));
    N = length(y);disp(N);
    % use matlab k means to build representation
    [idx,cl] = kmeans(label,k(ki),'emptyaction','singleton');
    x = zeros(N,k(ki));
    yy = [];

    % for each image
    for i = 1:N
    	% if there wasn't a problem in extracting sift features
    	if ~isempty(indices{i})
           yy = [yy;y(i)]; 
	   % do hisogram count for particular features belonging to the ith image
           x(i,:) = histc(idx(indices{i}),(1:k(ki)));
    	end
    end
    
    % generate data set for this sift setup
    partition(x,yy,k(ki));
end
end

% make the image grayscale if it isn't already
function I = convertg(I)
if size(I,3) ~= 1
    I = rgb2gray(I);
end
end


function partition(x,y,k)
n = max(unique(y));
train_index = floor(length(y) * .7);
train_x = x(1:train_index,:);
test_x = x(train_index + 1:end,:);
train_y = zeros(size(train_x,1),n);
test_y = zeros(size(test_x,1),n);
for i = 1:length(y)
    if i <= train_index
        train_y(i,y(i)) = 1;
    else
        test_y(i - train_index,y(i)) = 1;
    end
end
save(sprintf('imagedata_psift_%d.mat',k),'train_x','train_y','test_x','test_y');
end