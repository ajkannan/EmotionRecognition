function I_trimmed = isolateface(I, template)
num_passes = 8;
scale = 1.2;
template_resized = imresize(template,1 / scale);
[mt,nt] = size(template_resized);
max_index = zeros;
max_value = zeros;
best_mt = mt;
best_nt = nt;
C_size = zeros;
for i = 1:num_passes
    mt = floor(mt * scale);
    nt = floor(nt * scale);
    template_resized = imresize(template_resized,scale);
    if any(size(template_resized) > size(I))
        continue;
    end
    C = normxcorr2(template_resized, I);
    C_trimmed = C(mt - 1:end,nt - 1:end);
    current_max_value = max(max(C_trimmed));
    current_max_index = find(C_trimmed == current_max_value); % find max index in picture
    if current_max_value > max_value
        max_value = current_max_value;
        max_index = current_max_index;
        best_mt = mt;
        best_nt = nt;
        C_size = size(C_trimmed);
    end
end
[i1,i2] = ind2sub(C_size,max_index);
x = i1;
y = i2;
I_trimmed = I(x:x + best_mt,y:y + best_nt);
end
