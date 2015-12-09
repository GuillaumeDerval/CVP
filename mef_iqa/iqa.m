function iqa(source, output, reduce, outputImgFile, outputValueFile)

path = source;
possibleFiles = dir([path '/*']);
files = {};
for Index = 1:length(possibleFiles)
    baseFileName = possibleFiles(Index).name;
    [folder, name, extension] = fileparts(baseFileName);
    extension = upper(extension);
    if (strcmpi(extension,'.png') || strcmpi(extension,'.jpg') || strcmpi(extension,'.tif'))
        files = [files baseFileName];
    end
end

N = length(files);
if (N == 0)
    error('no files found');
    
end

% allocate memory
sz = size(imread(strcat(path,'/',files{1})));
r = floor(sz(1)*reduce);
c = floor(sz(2)*reduce);
I = zeros(r,c,3,N);

% read all files
for i = 1:N
    
    % load image
    filename = strcat(path,'/',files{i});
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

fI = imread(output);
fI = double(rgb2gray(fI));
[Q, Qs, QMap] = mef_ms_ssim(imgSeq, fI);

if ~isempty(outputImgFile)
    h = figure(1);
    subplot(2,2,1), imshow(fI/255), title([output, sprintf(' - %f', Q)], 'Interpreter','none');
    subplot(2,2,2), imshow(QMap{1}), title('quality map scale1');
    subplot(2,2,3), imshow(QMap{2}), title('quality map scale2');
    subplot(2,2,4), imshow(QMap{3}), title('quality map scale3');
    saveas(h, outputImgFile);
    close(h);
end

if ~isempty(outputValueFile)
    fileID = fopen(outputValueFile, 'w');
    fprintf(fileID, '%f', Q);
    fclose(fileID);
end

disp(Q);

exit(0);

end
        