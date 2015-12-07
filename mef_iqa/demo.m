clc,clear,close all;

%% change direction
prev_dir = pwd; file_dir = fileparts(mfilename('fullpath')); cd(file_dir);
addpath(genpath(pwd));

%% model calculation
Q = zeros(2,1); % test two fused images
imgSeqColor = uint8(load_images('./images/Mask',1));
[s1, s2, s3, s4] = size(imgSeqColor);
imgSeq = zeros(s1, s2, s4);
for i = 1:s4
    imgSeq(:, :, i) =  rgb2gray( squeeze( imgSeqColor(:,:,:,i) ) ); % color to gray conversion
end
fI1 = imread('./images/mask_2.png');
fI1 = double(rgb2gray(fI1));
[Q(1), Qs1, QMap1] = mef_ms_ssim(imgSeq, fI1);

fI2= imread('./images/mask_tm.png');
fI2 = double(rgb2gray(fI2));
[Q(2), Qs2, QMap2] = mef_ms_ssim(imgSeq, fI2);

Q

figure;
subplot(2,4,1), imshow(fI1/255), title('Mertens07');
subplot(2,4,2), imshow(QMap1{1}), title('quality map scale1');
subplot(2,4,3), imshow(QMap1{2}), title('quality map scale2');
subplot(2,4,4), imshow(QMap1{3}), title('quality map scale3');
subplot(2,4,5), imshow(fI2/255), title('lsaverage');
subplot(2,4,6), imshow(QMap2{1}), title('quality map scale1');
subplot(2,4,7), imshow(QMap2{2}), title('quality map scale2');
subplot(2,4,8), imshow(QMap2{3}), title('quality map scale3');
        