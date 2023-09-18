% This is VERY conceptual and not plausible for
% actual data analysis.
basedir = '/Users/jungwookim/github_primes/yangchogosu/HopfieldNet';
Xstruct = load(fullfile(basedir, 'data', 'subROIs.mat'));   Xcell = Xstruct.subROIs;
n_subs  = numel(Xcell);
n_nodes = unique(cellfun(@(x) size(x, 2), Xcell));

%% Weight constructions.
% Regularized partial correlation. 
cols         = 1:n_nodes;
W            = cell(n_subs, 1);
stds         = cell(n_subs, 1);
Wnew         = zeros(n_nodes, n_nodes, n_subs);
for iS = 1:n_subs
    X   = zscore(Xcell{iS});
    for i = 1:n_nodes
        xidx             = ~ismember(cols, i);
        L1_y             = X(:, i);
        L1_x             = X(:, xidx);
        fprintf('Running Lasso reg on %03d/%03d node --- %03d/%03d sub \n', i, n_nodes, iS, n_subs)
        out              = lasso_rocha(L1_y, L1_x);
        L1_B             = out.nbeta(end, :);
        % lambda is arbitrarily determined
        L1_yfit          = L1_x * L1_B';
        L1_errstd        = std(L1_y - L1_yfit);
        W{iS}(i, xidx)   = L1_B;
        stds{iS}(i, :)   = L1_errstd;
    end
end

for iS = 1:n_subs
    for ir = 1:n_nodes
        for ic = 1:n_nodes
            Wnew(ir, ic, iS) = W{iS}(ir, ic) * stds{iS}(ir) / stds{iS}(ic) ;
        end
    end
end

Wmat = mean(Wnew, 3); 
Wmat(logical(eye(size(Wmat)))) = 0;
% Is the convergence guaranteed? yes. (in the paper)
% so the Energy is not used for training...?


%%  One sample simulation.
% How is it different from just optimization process...? using fmincon...?
temperature = 0.41; % Interesting thing happens between 1 and 2.
n_iter      = 200;
clear Ats E_Ats
Efun    = @(x)(-x'*Wmat*x)/2;

% rng('default')
a0   = tanh(randn(n_nodes, 1)); 
for it = 1:n_iter
    if it == 1
        at = tanh(temperature * Wmat * a0); %  + normrnd(0, sigma_val, [size(a0) 1]);
    else 
        at = tanh(temperature * Wmat * at); % + normrnd(0, sigma_val, [size(a0) 1]);
    end
    Ats(:,it)    = at;
    E_Ats(:, it) = Efun(at);
end
plot(E_Ats); title('Energy level from single simulation') % This should converge.
%% Get attractor states
clear final_states
rng('default')
for iT = 1:1000
    a0   = tanh(randn(n_nodes, 1)); 
    for it = 1:n_iter
        if it == 1
            at = tanh(temperature * Wmat * a0); % + normrnd(0, sigma_val, [size(a0) 1]);
        else 
            at = tanh(temperature * Wmat * at); % + normrnd(0, sigma_val, [size(a0) 1]);
        end
    end
    final_states(:, iT) = at;
end

% I am not sure how they picked the attractor states... using k means
% clustering...
% get "kmeans" in canlabcore out of the search directory.
clear k_clust_meandists
test_clusts = 1:10;
clusters    = cell(numel(test_clusts), 1);
rng('default');
for i = test_clusts
    [clust_idx, clust, dists] = kmeans(final_states', i);
    k_clust_meandists(i) = mean(dists);
    clusters{i} = clust;
    clust_labels{i} = clust_idx;
end
plot(k_clust_meandists) % choose 2.
%% Visualize Attractors.
n_cluster = 2;
AttractorStates = clusters{2};
corr(AttractorStates') % same thing from the result is happening again. + - sign inverse.
obj        = fmri_data(which('Fan_et_al_atlas_r280.nii'));
clust_objs = cell(n_cluster, 1);

for iC = 1:n_cluster
    vis_temp      = obj;
    tobe_assigned = AttractorStates(iC, :);
    for iR = 1:280
        to_assign            = tobe_assigned(iR);
        intidx               = obj.dat == iR;
        vis_temp.dat(intidx) = to_assign;
    end
    clust_objs{iC} = vis_temp;
end
orthviews_multiple_objs(clust_objs)

%% Add stochastic updates.
n_stoch_iter = 1000;
sigma_val    = 0.001; % Had to try multiple params... no good...
rng('default')
clear Ats E_Ats
a0_t  = cellfun(@(x) mean(corr(x), 2), Xcell, 'UniformOutput', false);
a0    = mean(cell2mat(a0_t), 2); % initializing w/ mean connectome.
for it = 1:n_stoch_iter
    if it == 1
        at = tanh(temperature * Wmat * a0) + normrnd(0, sigma_val, [n_nodes 1]);
    else 
        at = tanh(temperature * Wmat * at) + normrnd(0, sigma_val, [n_nodes 1]);
    end
    Ats(:,it)    = at;
    E_Ats(:, it) = Efun(at);
end
subplot 211;
plot(E_Ats); title('Energy');

% Is it just random?
% y  = fft(randn(1, 1000));
y  = fft(zscore(E_Ats));
fs = 1;
f  = (0:length(y)-1)*fs / numel(y);
subplot 212; 
plot(f, abs(y)); title('Freq by Amp')

%% Visualized in PCs.
red = [1  .1  .1];
blk = [.1 .1 .1];
clear clrs;
for iC = 1:3
    clrs(:, iC) = linspace(red(iC),blk(iC), n_stoch_iter);
end

[~, sortidx] = sort(E_Ats);
[PCcoeff, PCscr, ~, ~, PCexpls] = pca(Ats', 'NumComponents', 5);
for i = 1:n_stoch_iter
    scatter(PCscr(i, 1),PCscr(i, 2), 20, 'MarkerEdgeColor', 'none',...
        'MarkerFaceColor', clrs(sortidx(i), :)); hold on;
end
xlabel(sprintf('PC1 (%.1f per)', PCexpls(1)));ylabel(sprintf('PC2 (%.1f per)', PCexpls(2)))
% The paper says it does multinomial regression... but it has no labels...?
% goin' to plot attractor states in the PC space.
%%%% NOTE: Lines in FCHN projections in the figure is not the same one as in here.
InPCspace = AttractorStates * PCcoeff;
for iA  = 1:size(InPCspace, 1)
    x_is = InPCspace(iA, 1);    y_is = InPCspace(iA, 2);
    line([0 2*x_is], [0 2*y_is]); hold on;
    text(2.1*x_is, 2.1*y_is, sprintf('state%d', iA));
end
% Plots of Energy states are bit different from the original paper...
% I think mine makes more sense.
% The figure in the paper shows that starting points are jumping around every where
% rather than converging into attractors...

%% Check proportion.
for iS = 1:n_subs
    singlesub = Xcell{iS};
    [n_time, ~] = size(singlesub);
    for iT = 1:n_time
        a0 = singlesub(iT, :)';
        for it = 1:50
            if it == 1
                at = tanh(temperature * Wmat * a0);
            else 
                at = tanh(temperature * Wmat * at);
            end
        end
        [~, maxidx] = max(corr(at, AttractorStates'));
        SubcellAttractors{iS}(:, iT) = maxidx;
    end
end
subplot 211;
histogram(clust_labels{4}) % clust_labels
subplot 212;
histogram(cat(2, SubcellAttractors{:}))

%% Draw dynamics
% Rest Dynamics
diffs   = cellfun(@diff, Xcell, 'UniformOutput', false);
diffPrj = cellfun(@(x) x * PCcoeff(:, 1:2), diffs, 'UniformOutput', false);
xmins   = cellfun(@(x) min(x(:, 1)), diffPrj);  xmaxs   = cellfun(@(x) max(x(:, 1)), diffPrj);
ymins   = cellfun(@(x) min(x(:, 1)), diffPrj);  ymaxs   = cellfun(@(x) max(x(:, 2)), diffPrj);

tol     = 0;
xminlim = min(xmins) - tol; xmaxlim = max(xmaxs) + tol;
yminlim = min(ymins) - tol; ymaxlim = max(ymaxs) + tol;
n_grids = 36;
xgrids  = linspace(xminlim, xmaxlim, n_grids);
ygrids  = linspace(yminlim, ymaxlim, n_grids);

for iS = 1:n_subs
    subPrj = diffPrj{iS};
    for iN1 = 1:n_grids-1
        for iN2 = 1:n_grids-1
            x_bot = xgrids(iN1); x_top = xgrids(iN1+1);
            y_bot = ygrids(iN2); y_top = ygrids(iN2+1);
            xidx = (x_bot <= subPrj(:, 1)) & (subPrj(:, 1) < x_top);
            yidx = (y_bot <= subPrj(:, 2)) & (subPrj(:, 2) < y_top);
            gridvecs{iS}(iN1, iN2, :) = [mean(subPrj(xidx, 1)), mean(subPrj(yidx, 2))];
        end
    end
end

mean_gridvec = nanmean(cat(4, gridvecs{:}), 4);
subplot 211;imagesc(mean_gridvec(:, :, 1));subplot 212;imagesc(mean_gridvec(:, :, 2));

hold on;
for iN1 = 1:n_grids-1
    for iN2 = 1:n_grids-1
        vecdir = squeeze(mean_gridvec(iN1, iN2, :));
        xp = mean([xgrids(iN1), xgrids(iN1+1)]);
        yp = mean([ygrids(iN2), ygrids(iN2+1)]);
        quiver(xp, yp ,vecdir(1), vecdir(2), 'Color', 'blue'); hold on;
    end
end

% Sample Dynamics
diffs   = {diff(Ats')};
diffPrj = cellfun(@(x) x * PCcoeff(:, 1:2), diffs, 'UniformOutput', false);
subPrj = diffPrj{1};
for iN1 = 1:n_grids-1
    for iN2 = 1:n_grids-1
        x_bot = xgrids(iN1); x_top = xgrids(iN1+1);
        y_bot = ygrids(iN2); y_top = ygrids(iN2+1);
        xidx = (x_bot <= subPrj(:, 1)) & (subPrj(:, 1) < x_top);
        yidx = (y_bot <= subPrj(:, 2)) & (subPrj(:, 2) < y_top);
        sample_gridvecs{1}(iN1, iN2, :) = [mean(subPrj(xidx, 1)), mean(subPrj(yidx, 2))];
    end
end

mean_gridvec = sample_gridvecs{:};
subplot 211;imagesc(mean_gridvec(:, :, 1));subplot 212;imagesc(mean_gridvec(:, :, 2));

hold on;
for iN1 = 1:n_grids-1
    for iN2 = 1:n_grids-1
        vecdir = squeeze(mean_gridvec(iN1, iN2, :));
        xp = mean([xgrids(iN1), xgrids(iN1+1)]);
        yp = mean([ygrids(iN2), ygrids(iN2+1)]);
        quiver(xp, yp ,vecdir(1), vecdir(2), 'Color', 'blue'); hold on;
    end
end

% Why this is the case?











%% Olds
% Wnew         = [];
% Corrs        = [];
% for iS = 1:n_subs
%     X = Xcell{iS};
%     % This takes a while.
%     Wnew(:, :, iS)  = partialcorr(X);
%     Corrs(:, :, iS) = corr(X);
% end
% load(fullfile(basedir, 'data', 'corrs.mat'));
% % save('corrs.mat', 'Wnew', 'Corrs');
% subplot 121;imagesc(mean(Wnew, 3)); colorbar;
% subplot 122;imagesc(mean(Corrs, 3)); colorbar; % looks quite different...
%     
% Wmat = mean(Wnew, 3); 
% Wmat(logical(eye(size(Wmat)))) = 0;
% % Instead of using full matrix, using top 90% connection to mimic sparsity.
% % Wmat_flat    = Wmat(:);
% % negidx       = Wmat_flat < 0;
% % absWmat_flat = abs(Wmat_flat);
% % [~, descidx] = sort(absWmat_flat, 'descend');
% % descidx(descidx < prctile(descidx, 50)) = 0;
% % absWmat_flat(negidx) = -absWmat_flat(negidx);
% % Wmat         = reshape(absWmat_flat .* descidx, n_nodes, n_nodes);
% Wmat         = zscore(Wmat, 0, 'all');
% Wmat(logical(eye(size(Wmat)))) = 0;
% % 'Wmat' is the weight matrix.
%%
