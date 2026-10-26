function tau = kendallTauMatrix(X)
    %KENDALLTAUMATRIX d x d Kendall's tau-b matrix from pairwise Knight's tau.
    %
    % Fills the symmetric tau-b matrix (unit diagonal) by calling kendallTauKnight on every column pair of X.
    % It equals corr(X, 'type', 'Kendall') to floating precision, ties included, at O(d^2 * n log^2 n) instead of O(d^2 * n^2), and it uses every row of X.
    % A degenerate or constant column yields NaN in its off-diagonal entries, as corr does, and the caller decides how to treat that (unitCorrelationFromTau reports it as nonfinite_kendall_tau).
    %
    % Inputs:
    %   X   - [n x d] data, samples down the rows; the caller removes non-finite rows, since kendallTauKnight returns NaN for a pair if any remain.
    % Outputs:
    %   tau - [d x d] symmetric Kendall tau-b matrix, diagonal 1.

    d = size(X, 2);
    tau = eye(d);
    for a = 1:d - 1
        for b = a + 1:d
            t = kendallTauKnight(X(:, a), X(:, b));
            tau(a, b) = t;
            tau(b, a) = t;
        end
    end
end
