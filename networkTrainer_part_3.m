clear
clc
close all

% path to all training and test data
dataPathname = 'D:\College\UWE Year 3\AdvancedMachineVision\Assignment\dataSet';

% load deep network and generate data store
anet = alexnet; % load AlexNet
reqImageSize = anet.Layers(1).InputSize; % image dimensions needed for AlexNet
dataStore = imageDatastore(dataPathname,... % generate a datastore variable
    'IncludeSubfolders',true,... % indicate data is split into folders
    'LabelSource','foldernames',... % use folder names as class labels
    'ReadFcn',@(f) repmat(imresize(imread(f),[227 227]),[1,1,3]),...
    'FileExtensions','.png'); % resize images to match that required by AlexNet
% the last part could have been:
%    'ReadFcn',@(f) repmat(imresize(imread(f),[227 227]),[1,1,3])); % resize images to match that required by AlexNet


% divide data into training and test
[imsTrain,imsTest] = splitEachLabel(dataStore,0.90,'randomized');

% modify layers
layers = anet.Layers;
clear anet % clear original network from memory (it's 250MB)
layers(end-2) = fullyConnectedLayer(10,'name','fc8'); % fully connected layer only needs ten nodes - one for each class (0-9)
layers(end) = classificationLayer('name','output'); % top layer

% set a few basic options
options = trainingOptions('sgdm',... % use gradient descent (with momentum)
    'Plots','training-progress',... % plot progress during training
    'ValidationData',imsTest,... % indicate which test data to use
    'MiniBatchSize',32,... % number of images used per mini-batch (don't worry too much about this for now, but keep low as to avoid slow training)
    'maxEpochs',10); % maximum number of epochs to use in training

% do the actual training
charNet = trainNetwork(imsTrain,layers,options);

% test performance
testPred = classify(charNet,imsTest); % classify entire validation set using final trained network
acc = sum(testPred == imsTest.Labels)/numel(imsTest.Labels); % get fraction of correct classifications
[cmap,clabel] = confusionmat(imsTest.Labels,testPred); % calculate confusion matrix
heatmap(clabel,clabel,cmap); % draw confusion matrix
title(sprintf('Test accuracy = %.1f%%',100*acc)) % make title equal to accuracy