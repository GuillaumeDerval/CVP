hdrImage = hdrread('chateau.hdr');
ldrImage = [imread('chateau0.jpg'),imread('chateau2.jpg')];
rgb = tonemap(hdr);
imshow(rgb);
%[Q, S, N, s_maps, s_local] = TMQI(hdrImage, ldrImage)