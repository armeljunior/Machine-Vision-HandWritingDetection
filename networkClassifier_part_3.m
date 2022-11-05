clc
close all
% variable not cleared as classifier network is required 

%% Images classifier

% file path of image to be classified
I = imread('C:\Users\james\Documents\3rd Year\Advanced Machine Vision\Assignment\dataSet\6\003.png');
% format image for AlexNet input
I = imresize(I,[227 227]);
I = repmat(im2uint8(I),1,1,3);
% calssify image using specified network 
label = classify(charNet,I);
% display the image with the classification
figure
imshow(I)
title(label)