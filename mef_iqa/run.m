imagename = 'mask';
reduce = 1;

%% model calculation
path = sprintf('../img/%s', imagename);
% find all JPEG or PPM files in directory
files = dir([path '/*']);
files = files(3:end);
N = length(files);
if (N == 0)
    error('no files found');
end

% allocate memory
sz = size(imread([path '/' files(1).name]));
r = floor(sz(1)*reduce);
c = floor(sz(2)*reduce);
I = zeros(r,c,3,N);

% read all files
for i = 1:N
    
    % load image
    filename = [path '/' files(i).name];
    im =double(imread(filename));
    if (size(im,1) ~= sz(1) || size(im,2) ~= sz(2))
        error('images must all have the same size');
    end
    
    % optional downsampling step
    if (reduce < 1)
        im = imresize(im,[r c],'bicubic');
    end
    if size(im,3)==1
    I(:,:,:,i) = cat(3,im,im,im);
    else
    I(:,:,:,i) = im;
    end
end


imgSeqColor = uint8(I);
[s1, s2, s3, s4] = size(imgSeqColor);
imgSeq = zeros(s1, s2, s4);
for i = 1:s4
    imgSeq(:, :, i) =  rgb2gray( squeeze( imgSeqColor(:,:,:,i) ) ); % color to gray conversion
end

imfiles = dir(sprintf('../output/%s*', imagename));

for k = 1:length(imfiles)
    disp(imfiles(k))
    fI = imread(strcat('../output/', imfiles(k).name));
    fI = double(rgb2gray(fI));
    [Q, Qs, QMap] = mef_ms_ssim(imgSeq, fI);
    figure(k);
    subplot(2,2,1), imshow(fI1/255), title(strcat(imfiles(k).name, sprintf(' - %f', Q)));
    subplot(2,2,2), imshow(QMap1{1}), title('quality map scale1');
    subplot(2,2,3), imshow(QMap1{2}), title('quality map scale2');
    subplot(2,2,4), imshow(QMap1{3}), title('quality map scale3');
    disp(Q);
end
        