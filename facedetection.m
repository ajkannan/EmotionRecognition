function I_trimmed = facedetection(I,T)
row_count = 300;
T_size = size(T);
final_size = (row_count / T_size(1)) * size(T);
I_trimmed = isolateface(I, T);
I_trimmed = imresize(I_trimmed, final_size);
end


