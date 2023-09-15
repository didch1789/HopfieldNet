addpath(genpath('~/github_primes'))
basedir = '/Users/jungwookim/github_primes/yangchogosu/HopfieldNet';
imgs    = sort_ycgosu(fullfile(basedir, 'data', '*crop*'));

clearvars -except basedir imgs
for ii = 1:numel(imgs)-1
     [~, PIs{ii}] = fileparts(imgs{ii});
     y            = double(imread(imgs{ii}));
     y_vec        = zscore(y(:));
     W(:, :, ii)  = y_vec * y_vec';
     y_vecs{ii}   = y_vec; 
end
Wnew = sum(W, 3);
Wnew = Wnew ./ numel(y_vec);
Wnew(logical(eye(size(Wnew)))) = 0;
%%
[rs, cs] = size(Wnew);
X0 = rand(1, rs);

for it = 1:100
    if it == 1
        Xt = X0 * Wnew;
    else
        Xt = Xt * Wnew;
    end
    Xt(Xt > 0) = 1;
    Xt(Xt < 0) = -1;
end
(abs(cellfun(@(x) corr(Xt', x), y_vecs)))
[~, maxidx]= max(abs(cellfun(@(x) corr(Xt', x), y_vecs)));
PIs(maxidx)









