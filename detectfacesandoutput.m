function [trcrop,tscrop] = detectfacesandoutput()
load Images/images.mat;
template = imread('template.jpg');
trcrop = detection(trimage,template);
tscrop = detection(tsimage,template);
save('cropped.mat','trcrop','tscrop','trlabel','tslabel');
end

function crop = detection(images,template)
crop = cell(length(images),1);
for i = 1:length(images)
    try
        crop{i} = facedetection(imresize(images{i},[490,640]),template);
    catch
    end
end
end
