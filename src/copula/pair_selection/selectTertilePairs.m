% SELECTTERTILEPAIRS Deterministic choice of representative channel pairs per |tau| tertile.
%
% [sel_idx, band_of] = selectTertilePairs(abs_tau, n_per_band) splits the pairs into three dependence-strength bands by the tertiles of |tau| and, within each non-empty band, returns the pairs whose |tau| is nearest that band's median |tau|.
% The rule uses no random numbers, so the chosen pairs are the same on every run.
%
% It chooses the few pairs shown in the d = 2 copula-density figure, a communication aid; it selects which pairs to display, not any reported number.
%
% Inputs:
%   abs_tau     - vector of |Kendall tau| over all candidate pairs (length n_pairs).
%   n_per_band  - (optional) how many pairs to return per band (default 1), taken in order of increasing distance to the band median, with ties broken toward the lowest pair index.
%
% Outputs:
%   sel_idx  - column vector of selected pair indices into abs_tau, low-|tau| band first; its length is at most 3 * n_per_band, fewer when a band is empty or smaller than n_per_band.
%   band_of  - [n_pairs x 1] band label of every pair: 1 = |tau| <= t1, 2 = t1 < |tau| <= t2, 3 = |tau| > t2, with [t1 t2] the tertiles.

function [sel_idx, band_of] = selectTertilePairs(abs_tau, n_per_band)

    if nargin < 2 || isempty(n_per_band)
        n_per_band = 1;
    end
    validateattributes(abs_tau, {'numeric'}, {'vector', 'real', 'nonnegative'}, mfilename, 'abs_tau', 1);
    validateattributes(n_per_band, {'numeric'}, {'scalar', 'integer', 'positive'}, mfilename, 'n_per_band', 2);

    abs_tau = abs_tau(:);
    n_pairs = numel(abs_tau);

    % Tertile edges of |tau| and the band of every pair.
    tertile_edges = quantile(abs_tau, [1 / 3, 2 / 3]);
    band_of = ones(n_pairs, 1);
    band_of(abs_tau > tertile_edges(1) & abs_tau <= tertile_edges(2)) = 2;
    band_of(abs_tau > tertile_edges(2)) = 3;

    sel_idx = zeros(0, 1);
    for b = 1:3
        idx = find(band_of == b);
        if isempty(idx)
            continue
        end
        band_median = median(abs_tau(idx));
        % Nearest to the band median first; sort is stable, so equal distances keep ascending index order and ties go to the lowest index.
        [~, order] = sort(abs(abs_tau(idx) - band_median));
        take = idx(order(1:min(n_per_band, numel(idx))));
        sel_idx = [sel_idx; take(:)]; %#ok<AGROW>
    end
end
