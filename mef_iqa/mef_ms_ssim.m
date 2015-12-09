function [oQ, Q, qMap] = mef_ms_ssim(imgSeq, fI, window)

%- imgSeq: source sequence in [0-255] grayscale.
%- fI: the MEF image to compare in [0-255] grayscale.
%- window: local window for statistics. default widnow is 
%  Gaussian given by window = fspecial('gaussian', 11, 1.5);
%            
%- oQ: The overall quality score of the MEF image.
%   Q: The quality scores in each scale.
%  qMap: The quality maps of the MEF image in each scale. 
%
% usage: [oQ, Q, qMap] = mef_ms_ssim(imgSeq, fI);

if (~exist('window', 'var'))
   window = fspecial('gaussian', 11, 1.5);
end

[H, W] = size(window);
level = 3;
weight = [0.0448  0.2856  0.3001]'; 
weight = weight / sum(weight);

[s1, s2, s3] = size(imgSeq);
minImgWidth = min(s1, s2)/(2^(level-1));
maxWinWidth = max(H, W);

if (minImgWidth < maxWinWidth)
   oQ = -Inf;
   Q = -Inf;
   qMap = Inf;
   return;
end

imgSeq = double(imgSeq);
fI = double(fI);
downsampleFilter = ones(2)./4;
Q = zeros(level,1);
qMap = cell(level,1);
if level == 1
    [Q, qMap] = mef_ssim(imgSeq, fI, window);
    oQ = Q;
    return;
else
    for l = 1 : level - 1
        [Q(l), qMap{l}] = mef_ssim(imgSeq, fI, window); 
        imgSeqC = imgSeq;
        clear imgSeq;
        for i = 1:s3
            rI = squeeze(imgSeqC(:,:,i));
            dI = imfilter(rI, downsampleFilter, 'symmetric', 'same');
            imgSeq(:,:,i) = dI(1:2:end, 1:2:end);
        end
        dI = imfilter(fI, downsampleFilter, 'symmetric', 'same');
        clear fI;
        fI = dI(1:2:end, 1:2:end);
    end
    % the coarsest scale
    [Q(level), qMap{level}] = mef_ssim(imgSeq, fI, window);
    Q = Q(:);
    oQ = prod(Q.^weight);
end

