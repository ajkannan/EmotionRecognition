function loaddata()
opts.eigenedges = 0;
opts.face = 1;
opts.median = 0;
opts.normal = 0;
opts.minimize = 0;
opts.surf = 1;
y = load('Labels/emotion_labels.txt');
c = textscan(fopen('Labels/associated_pic_filenames.txt'),'%s');
im = imread(strcat(['../.' char(c{1}(1))]));
s = size(im);
if opts.face
   if opts.minimize
      x = zeros(length(y),30*35);
   else
      x = zeros(length(y),300*348);
   end
   template = imread('Labels/template.jpg');
else
   x = zeros(length(y),numel(im));
end
for i = 1:length(c{1})
    fprintf('image id: %d\n',i);
    I = imread(strcat(['../.' char(c{1}(i))]));
    if size(I,3) == 3
        I = rgb2gray(I);
    end
    I = imresize(I,s);
    if opts.face
       try
       I = facedetection(I,template);
       catch
       end
       if opts.minimize
       I = imresize(I,.1);
       end
       % keyboard;
    end
    if opts.median
       I = medfilt2(I);
    end
    if opts.eigenedges
        I = smmeedges(im2double(I),.005,5);
    end
    if opts.surf
       points = detectSURFFeatures(I);
       I = I * 0;
       for j = 1:length(points.Location(:,1))
       	   I(round(points.Location(j,2)),round(points.Location(j,1))) = 1;
       end
    end
    x(i,:) = I(:);
end
n = max(unique(y));
train_index = floor(length(y) * .7);
train_x = x(1:train_index,:);
test_x = x(train_index + 1:end,:);

if opts.normal
   m = mean(train_x);
   sd = std(train_x);
   train_x = bsxfun(@rdivide,bsxfun(@minus,train_x,m),sd);
   test_x = bsxfun(@rdivide,bsxfun(@minus,test_x,m),sd);
end

train_y = zeros(size(train_x,1),n);
test_y = zeros(size(test_x,1),n);
for i = 1:length(y)
    if i <= train_index
        train_y(i,y(i)) = 1;
    else
        test_y(i - train_index,y(i)) = 1;
    end
end
save('imagedata.mat','train_x','train_y','test_x','test_y');
end