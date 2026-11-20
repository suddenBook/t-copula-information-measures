function diff = tInformationDifferences(d, p)
    % TINFORMATIONDIFFERENCES Closed-form t-minus-Gaussian information differences.
    %
    % diff = tInformationDifferences(d, p) returns, in nats, the amount by which each information measure of a d-variate Student-t copula with shape parameter p = 1/nu exceeds the Gaussian value at the same correlation matrix R.
    % The differences do not depend on R: the Gaussian level cancels and only the copula-entropy shape term kappa_t survives, so they are functions of (d, p) alone.
    %
    % With kappa_d = kappa_t(d, nu) and kappa_dm1 = kappa_t(d-1, nu), which is the exact leave-one-out shape term because the t family is closed under marginalisation with the same nu:
    %
    %   MI  = -kappa_d                     (d = 2 only; NaN at d >= 3)
    %   TC  = -kappa_d
    %   DTC = d * kappa_dm1 - (d-1) * kappa_d
    %   O   = TC - DTC = (d-2) * kappa_d - d * kappa_dm1
    %
    % kappa_t(1, nu) = h_1 - h_1 = 0 exactly, so kappa_dm1 is exactly 0 at d = 2 and DTC = TC there, as it must be for a pair.
    %
    % p may be an array of any shape, and every field except d has that shape.
    % At p = 0 every shape term and therefore every difference is exactly 0.
    % kappa <= 0 with equality only at p = 0, so MI and TC are non-negative; that bound does not fix the signs of DTC and O, which are not to be read as losses.
    %
    % Inputs:
    %   d - dimension, integer >= 2.
    %   p - shape parameters p = 1/nu, real, finite, >= 0, any shape.
    % Output:
    %   diff - struct with d, p, nu, kappa_d, kappa_dm1, MI, TC, DTC, O.

    arguments
        d (1, 1) double {mustBeInteger, mustBeGreaterThanOrEqual(d, 2)}
        p double {mustBeReal, mustBeFinite, mustBeNonnegative}
    end

    ce = tCopulaEntropy(d, p);
    kappa_d = ce.kappa;
    kappa_dm1 = ce.kappa_margin;

    TC = -kappa_d;
    DTC = d * kappa_dm1 - (d - 1) * kappa_d;
    O = TC - DTC;
    if d == 2
        MI = TC;
    else
        MI = nan(size(p));
    end

    diff = struct( ...
                  'd', d, ...
                  'p', p, ...
                  'nu', 1 ./ p, ...
                  'kappa_d', kappa_d, ...
                  'kappa_dm1', kappa_dm1, ...
                  'MI', MI, ...
                  'TC', TC, ...
                  'DTC', DTC, ...
                  'O', O);
end
