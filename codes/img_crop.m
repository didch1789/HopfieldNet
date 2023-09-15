% cropping img.
addpath(genpath('~/github_primes'))
basedir = '/Users/jungwookim/github_primes/yangchogosu/HopfieldNet';
imgs   = sort_ycgosu(fullfile(basedir, 'data', '*jpg'));
imgs   = imgs(~contains(imgs, 'crop'));
imsize = [40 40];

for ii = 1:numel(imgs)
    [~, who_isit] = fileparts(imgs{ii});
    img_is     = rgb2gray(imread(imgs{ii}));
    [row, col] = size(img_is);
    row_cent = round(row/2);
    col_cent = round(col/2);
    int_idx1 = row_cent-floor(imsize(1)/2):row_cent+floor(imsize(1)/2)-1;
    int_idx2 = col_cent-floor(imsize(2)/2):col_cent+floor(imsize(2)/2)-1;
    img_cropped = img_is(int_idx1, int_idx2);
    figure(ii);imshow(img_cropped)
    imwrite(img_cropped, fullfile(basedir, 'data', sprintf('%s_crop.jpg', who_isit)));
end