function loadajayhogdata()
y = load('Labels/emotion_labels.txt');
e = load('cropped_eyes.mat');
c = textscan(fopen('Labels/associated_pic_filenames.txt'),'%s');

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
    
    dm = extracthog(Im)

    x(i,1:81) = dm;
    index(i) = 1;
    catch er
        disp(er)
    end
end
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
save('imagedata_amouthhog.mat','train_x','train_y','test_x','test_y');
end
