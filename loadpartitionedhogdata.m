function loadpartitionedhogdata()
y = load('Labels/emotion_labels.txt');
c = textscan(fopen('Labels/associated_pic_filenames.txt'),'%s');
% s = size(imread(strcat(['../.' char(c{1}(1))])));

N = length(c{1});
index = zeros(N,1);
x = zeros(N,81*2);
for i = 1:N
    try
    fprintf('image id: %d\n',i);
    detector = buildDetector();
    I = imread(strcat(['../.' char(c{1}(i))]));
    bbox = detectFaceParts(detector,I);
    Im = convertg(imcrop(I,bbox(end,13:16)));
    Ie = convertg(imcrop(I,bbox(end,5:8)));
    figure;imshow(Im);
    figure;imshow(Ie);
    pause(1);
    
    dm = extracthog(Im)
    de = extracthog(Ie)
    
    x(i,1:81) = dm;x(i,82:end) = de;
    index(i) = 1;
    catch e
        disp(e)
    end
    % figure;imshow(bbimg);
    % for j = 1:size(bbfaces,1)
    %     figure;imshow(bbfaces{j});
    % end
    % keyboard;
    % [~,descriptors,~] = sift(strcat(['../.' char(c{1}(i))]));
    % imshow(image);pause(.1);
    % indices{i} = size(label,1) + (1:size(descriptors,1));
    % label = [label;descriptors];
    % if i == 52
    %     keyboard;
    % end
end
% N = length(y);disp(N);
% [idx,cl] = kmeans(label,k);
% x = zeros(N,k);
% for i = 1:N
%     if ~isempty(indices{i})
%         x(i,:) = histc(idx(indices{i}),(1:k));
%     end
%     if i == 52
%     keyboard;
%     end
% end
% keyboard;
partition(x(logical(index),:),y(logical(index)));
end

function I = convertg(I)
if size(I,3) ~= 1
    I = rgb2gray(I);
end
end

function partition(x,y)
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
save('imagedata_phog.mat','train_x','train_y','test_x','test_y');
end