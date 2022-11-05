clear
clc
close all

%% Averaging values

% set the number of runs for  the average
averagingRuns = 10;
% store accuracy of each run
pctCorrect = nan(1, averagingRuns);

for runNum = 1:averagingRuns
    % clear values between test runs (except averaging data)
    clearvars -except averagingRuns pctCorrect runNum
    
    maxNum = 9; % maximum numeral to recognise
    testFrac = 0.2; % fraction of images used for testing 
    k = 10; % k value
    % setting up random number generator
    seed = clock; 
    seed = round(seed(1,6)*100,0);
    rng(seed);
    % get list of image file names (should be same in each folder)
    names = dir('/Users/james/Documents/3rd Year/Advanced Machine Vision/Assignment/dataSet/0/*.png');
    % total number of images for each class (numeral in this case)
    NClass = length(names);
    % number of training and testing images
    NTest = randperm(NClass,round(NClass*testFrac)); % Permutation to prevent repeating numbers
    NTrain = NClass - NTest;
    % initialise vectors that will contain metrics 
    classLabel = nan(NClass*(maxNum+1),1);
    % indicate which images are for test only
    testOnly = false(size(classLabel));
    % define matrix to store hu moments
    huValues = nan(7, length(NTest));

    %% Compute all metrics
    
    ind = 0;
    for num = 0:maxNum % for each class
        % get path for images in current class (numeral)
        pathToIms = fullfile('imageData',num2str(num));
        % get list of image filenames for current class
        imFilenames = dir(fullfile(pathToIms,'*.png'));
        for imNo = 1:NClass % for each image of current class
            ind = ind + 1;
            % read next image
            im = imread(fullfile(pathToIms,imFilenames(imNo).name));
            % calculate and store hu moments
            huValues(:,ind) = generateHu(im);
            % store corresponding class
            classLabel(ind) = num;
            % if image has been selected for testing
            if ismember(imNo, NTest)
                testOnly(ind) = true;
            end
        end
    end


    %% Build the classifier classifier 

    huValues = transpose(huValues);
    % construct a single matrix containing all metrics for training
    allMetricsTrain = [huValues((classLabel<=maxNum & ~testOnly),1) huValues((classLabel<=maxNum & ~testOnly),2) huValues((classLabel<=maxNum & ~testOnly),3) huValues((classLabel<=maxNum & ~testOnly),4) huValues((classLabel<=maxNum & ~testOnly),5) huValues((classLabel<=maxNum & ~testOnly),6) huValues((classLabel<=maxNum & ~testOnly),7)];
    % construct a single matrix containing all metrics for testing
    allMetricsTest = [huValues((classLabel<=maxNum & testOnly),1) huValues((classLabel<=maxNum & testOnly),2) huValues((classLabel<=maxNum & testOnly),3) huValues((classLabel<=maxNum & testOnly),4) huValues((classLabel<=maxNum & testOnly),5) huValues((classLabel<=maxNum & testOnly),6) huValues((classLabel<=maxNum & testOnly),7)];
    % vector showing correct class for each image in test dataset (will eventually be used for evaluation)
    classTest = classLabel(classLabel<=maxNum & testOnly);

    % use built in function to build Kd-tree
    kdTreeModel = KDTreeSearcher(allMetricsTrain);

    %% Evaluate accuracy

    preds = zeros(size(classTest));
    classTrain = classLabel(classLabel<=maxNum & ~testOnly);
    for testNo = 1:length(classTest) % for each test image
        [inds,dists] = knnsearch(kdTreeModel,allMetricsTest(testNo,:),'k',k);
        preds(testNo) = mode(classTrain(inds)); % predict class
    end
    % find correct prediction
    numCorrect = sum(preds==classTest);
    pctCorrect(runNum) = 100*numCorrect/length(classTest);
    % display accuracy results
    fprintf('Run %d: Accuracy = %d of %d (%.1f%%)\n',runNum,numCorrect,length(classTest),pctCorrect(runNum));
end

%% Plot average

% calculate mean accuracy for all test runs
totalMean = sum(pctCorrect)/averagingRuns;

% plot mean accuracy
figure(1)
hold on
plot(pctCorrect, 'k');
yline(totalMean, '--r', 'Total mean');
title('Average Accuracy')
xlabel('Test run')
ylabel('Mean accuracy (%)')
hold off
% print to terminal
fprintf('Mean Accuracy = %.1f%%\n',totalMean);

% function to calculate Hu moments
function huMo = generateHu(im) 

M00 = sum(sum(im)); 
M10 = 0; 
M01 = 0; 
[row,col] = size(im);

for x = 1:row 
    for y = 1:col 
        % moments
        M10 = M10+x*im(x,y); 
        M01 = M01+y*im(x,y); 
    end 
end

N20 = 0; N02 = 0; N11 = 0; N30 = 0; N12 = 0; N21 = 0; N03 = 0;

for x = 1:row 
    for y = 1:col 
        % central moments
        N20 = N20+x^2*im(x,y); 
        N02 = N02+y^2*im(x,y); 
        N11 = N11+x*y*im(x,y); 
        N30 = N30+x^3*im(x,y); 
        N03 = N03+y^3*im(x,y); 
        N12 = N12+x*y^2*im(x,y); 
        N21 = N21+x^2*y*im(x,y); 
    end 
end 
  % normalised moments
  N20 = N20/M00^2; 
  N02 = N02/M00^2; 
  N11 = N11/M00^2; 
  N30 = N30/M00^2.5; 
  N03 = N03/M00^2.5; 
  N12 = N12/M00^2.5; 
  N21 = N21/M00^2.5; 
  % hu equations    
  Hu1 = N20 + N02;                      
  Hu2 = (N20-N02)^2 + 4*(N11)^2; 
  Hu3 = (N30-3*N12)^2 + (3*N21-N03)^2;  
  Hu4 = (N30+N12)^2 + (N21+N03)^2; 
  Hu5 = (N30-3*N12)*(N30+N12)*((N30+N12)^2-3*(N21+N03)^2)+(3*N21-N03)*...
      (N21+N03)*(3*(N30+N12)^2-(N21+N03)^2); 
  Hu6 = (N20-N02)*((N30+N12)^2-(N21+N03)^2)+4*N11*(N30+N12)*(N21+N03); 
  Hu7 = (3*N21-N03)*(N30+N12)*((N30+N12)^2-3*(N21+N03)^2)+(3*N12-N30)*...
      (N21+N03)*(3*(N30+N12)^2-(N21+N03)^2); 
 % return hu moments
 huMo = [Hu1 Hu2 Hu3 Hu4 Hu5 Hu6 Hu7]; 
end


