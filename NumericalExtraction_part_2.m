clear
clc
close all

upData = 'Is input image binarized? Y[0]/N[1]: ';
answer = input(upData);

%% 1: Read and display raw image
% skips binarization step if image is already binary 
if answer
    im = imread('D:\College\UWE Year 3\AdvancedMachineVision\Assignment\dataSet\0\001.png');
    figure
    subplot(131)
    imshow(im)
    title('RGB');
else
    im = imread('D:\College\UWE Year 3\AdvancedMachineVision\Assignment\dataSet\0\001.png');
end

ImWidth = size(im,2);
ImHeight = size(im,1);

%% 2: Convert to grey and display
if answer
    im = rgb2gray(im);
    subplot(132)
    imshow(im)
    title('Greylevel')
end

%% 3: Binarise and display
if answer
    im = im < 220; % set im to TRUE for all pixels below the threshold (ink) and FALSE otherwise
    subplot(133)
    imshow(im)
    title('Binary')
end

%% 4: Plot number of TRUE (ink) pixels for each column of image
chk = sum(im,1);
figure
subplot(211)
plot(chk)
title('Sum of number of ink pixels for each column')

%% 5: Determine which columns have at least one pixel that is TRUE (ink) and plot
chk2 = chk > 0;
subplot(212)
plot(chk2)
title('Flag to show if any pixels in a given column are ink')

%% 6: Get column numbers where ink changes from being not present to present
t = 1;
for i = 2:ImWidth
    if chk2(i) == 1 && chk2(i - 1) == 0
        xStart(t) = i-1;    %Finds left of integer
        t = t + 1;
    end
end
t = 1;
for i = 2:ImWidth
    if chk2(i) == 0 && chk2(i - 1) == 1
        xStop(t) = i;       %Finds right of integer
        t = t + 1;
    end
end

%% 7: Crop image into individual digits
figure
for i = 1:(t-1)
    imCrop{i} = im(1:ImHeight,xStart(i):xStop(i));  %Crops image using found edges of character
end
for i = 1:(t-1)
    subplot(1,(t-1),i);
    imshow(imCrop{i})
end

%% 8: Find top and bottom coordinates of each number in images
for i = 1:(t-1)
    chk = sum(imCrop{i},2);
    chk2 = chk > 0;
    
    for j = 2:ImHeight
        if chk2(j) == 1 && chk2(j - 1) == 0
            yStart(i) = j - 1;          %Finds top of integer
        end
    end

    for j = 2:ImHeight
        if chk2(j) == 0 && chk2(j - 1) == 1
            yStop(i) = j;               %Finds bottom of integer
        end
    end
    
end

%% 9: Crop out above and below number
for i = 1:(t-1)
    imCrop{i} = im(yStart(i):yStop(i),xStart(i):xStop(i)); %Crops images again using found edges
end

%% 10: Altering resulting images to fit into kNN and AlexNet algorithms
figure
for i = 1:(t-1)
    subplot(1,(t-1),i);
    imshow(imCrop{i})
end

for i = 1:(t-1)
    ImWidth = size(imCrop{i},2);
    ImHeight = size(imCrop{i},1);

    ImWidth = round((227 - ImWidth)/2,0);
    ImHeight = round((227 - ImHeight)/2,0);

    imCrop{i} = padarray(imCrop{i},[ImHeight ImWidth], 0);  %Adds black pixels around image to expand
    
    %Ensures images are 227 x 227 pixels
    if size(imCrop{i},1) == 228
        imCrop{i}(1,:) = [];
    end
    if size(imCrop{i},2) == 228
        imCrop{i}(:,1) = [];
    end
    
    imCrop{i} = repmat(im2uint8(imCrop{i}),235,235,1);
end



