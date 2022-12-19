close all
clear
clc

% LOAD TRAINED MODEL
file = load("trained_coin_classifier_4_classes_uanglama_10_minibatch.mat");
newnet = file.netTransfer;

% PREDICT
img_path = "../img/contoh.jpeg";
fprintf('Jumlah uang = ');
total_uang = countCoins(img_path, newnet);
disp(total_uang);

% FUNCTIONS
function total = countCoins(img_path, newnet)
    % READ IMAGE
    img = imread(img_path);
    figure();
    imshow(img);
    title("Citra masukan");
    
    % TO GRAYSCALE
    gr = rgb2gray(img);
    figure;
    imshow(gr);
    title("Citra grayscale");

    % IMAGE SMOOTHING
    blurred = imgaussfilt(gr,2);
    figure;
    imshow(gr);
    title("Citra yang sudah di smooth");
    
    % CHANGE IMAGE TO BINARY
    bw = imbinarize(blurred); %// Read the image
    figure;
    imshow(bw);
    title("Citra biner");
    
    % MORPHOLOGICALLY CLOSE IMAGE
    se = strel('disk',100);
    closed_img = imclose(bw, se);
    figure();
    imshow(closed_img);
    title("Citra biner yang sudah di-morphologically close");
    
    % FILL THRE AREAS SURROUNDED BY CONNECTED EDGES
    BW_filled = imfill(closed_img,'holes');
    figure();
    imshow(BW_filled);
    title("Citra biner yang sudah diisi");
    
    % FIND ALL OBJECT'S BOUNDARY 
    boundaries = bwboundaries(BW_filled);
    props = regionprops(BW_filled, 'BoundingBox');

    % PLOT ALL OBJECT
    L = length(boundaries);

    % PLOT BOUNDARIES AND REGIONPROPS
    figure();
    imshow(BW_filled);
    title("Boundary dan bounding box yang terdeteksi");
    hold on;
    for k=1:L
        b = boundaries{k};
        plot(b(:,2),b(:,1),'g','LineWidth',5);
        thisBB = props(k).BoundingBox;
        rectangle('Position', thisBB, 'EdgeColor', 'r', 'LineWidth',5);
    end

    
    % GET SEGMENTED IMAGES + PREDICT
    amounts = [];
    
    for k=1:L
        % Get bounding box for object
        thisBB = props(k).BoundingBox;
    
        % Get a cropped image.
        croppedImage = imcrop(img, thisBB);
    
        % Resize cropped image
        im = imresize(croppedImage, [227 227]);
    
        % Classify what coin is it
        [YPred,scores] = classify(newnet, im);
    
        % Get amount
        amount = getAmount(string(YPred));
        amounts = [amounts amount];
    
        % Show object image
        figure();
        imshow(croppedImage);
        title(amount, 'FontSize',15);
    end

    % COUNT TOTAL MONEY
    total = sum(amounts);
end

function amount = getAmount(label)
    if (label == "duaratus_b")
        amount = 200;
    elseif (label == "limaratus_b")
        amount = 500;
    elseif (label == "seratus_b")
        amount = 100;
    else
        amount = 1000;
    end
end