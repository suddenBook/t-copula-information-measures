function plan = tCopulaQuantilePlan(U)
    %TCOPULAQUANTILEPLAN Shape-independent preparation of a pseudo-observation matrix.
    %
    % plan = tCopulaQuantilePlan(U) precomputes everything the t-copula log-likelihood needs from the pseudo-observations U (n x d, strictly in (0, 1)) that does not depend on the shape parameter, so that each likelihood evaluation calls the quantile function once per distinct two-sided tail probability instead of once per matrix entry.
    % For rank pseudo-observations r/(n+1) the ranks r and n+1-r share the tail probability 2*min(u, 1-u), so the number of distinct arguments is about n/2 whatever the dimension.
    %
    % Outputs:
    %   plan - struct with fields
    %     n, d                 matrix size
    %     q_unique             m x 1 ascending distinct values of 2*min(U, 1-U)
    %     q_index              n x d uint32 index into q_unique
    %     negate               n x d logical, true where u < 1/2 (the quantile is negative)
    %     q_count              m x 1 multiplicity of each distinct value
    %     num_distinct         m
    %     min_tail_probability smallest one-sided tail probability min(u, 1-u) present

    arguments
        U (:, :) double {mustBeReal, mustBeFinite}
    end
    [n, d] = size(U);
    assert(d >= 2, 'tCopulaQuantilePlan:InvalidDimension', ...
        'U must have at least two columns; got %d.', d);
    assert(all(U(:) > 0 & U(:) < 1), 'tCopulaQuantilePlan:InvalidPseudoObservations', ...
        'Pseudo-observations must lie strictly inside (0, 1); got min %.3g and max %.3g.', ...
        min(U(:)), max(U(:)));

    q = 2 * min(U, 1 - U);
    [qUnique, ~, qIndex] = unique(q(:));
    % 1 - r/(n+1) and (n+1-r)/(n+1) differ by rounding, so the mirror ranks would otherwise not merge; consecutive values closer than a few ulps are one argument (genuine neighbours are at least 1/(n+1) apart).
    keep = [true; diff(qUnique) > 8 * eps(qUnique(2:end))];
    groupId = cumsum(keep);
    qUnique = qUnique(keep);
    qIndex = groupId(qIndex);
    plan = struct( ...
        'n', n, ...
        'd', d, ...
        'q_unique', qUnique, ...
        'q_index', reshape(uint32(qIndex), n, d), ...
        'negate', U < 0.5, ...
        'q_count', accumarray(qIndex, 1, [numel(qUnique), 1]), ...
        'num_distinct', numel(qUnique), ...
        'min_tail_probability', qUnique(1) / 2);
end
