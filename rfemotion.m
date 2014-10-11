% load the data from the generated image representations
data = dir('Data/imagedata_amouthhog.mat'); 

% the mininum number of examples in the leaves of the random forest
ml = 5; 

% binary variable for correcting the unbalanced class problem
unbalanced = 1;
if unbalanced
   % use more trees if unbalanced
   trees = 500:100:1000;
else
   trees = 20:20:200;
end

% store results in "a" for each image representation and the number of trees in the forest
a = zeros(length(data),length(trees));

% for each representation
for d = 1:length(data)
    % load the data
    load(['Data/' data(d).name]);
    fprintf('loaded %s\n',data(d).name);
    
    % normalize the data
    m = mean(train_x);
    sd = std(train_x);sd(1) = mean(sd);sd(end) = mean(sd);
    train_x = bsxfun(@rdivide,bsxfun(@minus,train_x,m),sd);
    test_x = bsxfun(@rdivide,bsxfun(@minus,test_x,m),sd);
    
    % reformat the target variables to be a column vector (instead of binary matrix)
    [~,train_y] = max(train_y,[],2);
    [~,test_y] = max(test_y,[],2);

    % for each value indicating the number of trees in the forest
    for i = 1:length(trees) 
    	% start a timer
	tic; 
    	fprintf('Using unbalanced: %d\n',unbalanced);
	fprintf('Using trees: %d\n',trees(i));

	% if correcting for unbalanced classes, train a random forest using the rusboost algorithm
    	if unbalanced
	   t = ClassificationTree.template('minleaf',ml);
	   ens = fitensemble(train_x,train_y,'RUSBoost',trees(i),t,'LearnRate',0.1);
	else
	   % otherwise just train a vanilla random forest
	   ens = fitensemble(train_x,train_y,'AdaBoostM2',trees(i),'Tree');disp(ens);
	end
	toc;

	% predict class labels for the training data set
        res = predict(ens,test_x);
	
	% generate confusion matrix
	C = confusionmat(test_y,res);
	fprintf('The confusion matrix:\n');disp(C);
        acc = sum(res == test_y) / length(res);
        a(d,i) = acc;
        fprintf('Random forest accuracy: %.2f\n',acc);

	% calculate precision and recall
	prec = (diag(C) ./ sum(C)');
	recl = (diag(C) ./ sum(C,2));
	prec(isnan(prec)) = 0;prec = mean(prec);
	recl(isnan(recl)) = 0;recl = mean(recl);

	% calculate F1 score as harmonic mean of precision and recall
	F1 = 2 * ((prec * recl) / (prec + recl));
	fprintf('Random forest F1: %.2f\n\n\n',F1);
    end
end
fprintf('*** The maximum accuracy: %.2f\n',max(max(a)));
save('rfemotion.mat','a');