classdef tCopulaSampleRadialTest < matlab.unittest.TestCase
    %TCOPULASAMPLERADIALTEST The sampler's radial law, checked against F(d, nu).
    %
    % The dependence check (the Kendall sine relation) and the marginal-moment check (kurtosis) on tCopulaSample do not see the radial part, which is where the degrees of freedom live.
    % If the chi-square mixing variate were drawn with the wrong parameter or the wrong scale, the copula would still have the right correlation and heavy-ish margins.
    %
    % For X ~ t_d(R, nu) the Mahalanobis form X' R^-1 X / d is exactly F(d, nu), so a Kolmogorov-Smirnov test against that fully specified distribution is an external reference for the sampler's radial law.
    % Each check requires both that kstest does not reject at its default 5% level and that the p-value exceeds SignificanceLevel (0.01), so the 5% level binds: a correct sampler fails it for about one seed in twenty, and the seeds here are fixed.

    properties (Constant)
        NumDraws = 50000
        SignificanceLevel = 0.01
    end

    methods (TestClassSetup)
        function addSourceTree(testCase)
            sourceTree = fullfile( ...
                fileparts(fileparts(mfilename('fullpath'))), 'src');
            assert(isfolder(sourceTree), ...
                'tCopulaSampleRadialTest:SourceTreeNotFound', ...
                'expected this file under <repo>/tests/; got %s', ...
                mfilename('fullpath'));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                sourceTree, IncludingSubfolders=true));
        end
    end

    methods (Test)
        function mahalanobisFormFollowsTheFDistribution(testCase)
            testCase.assumeTrue(exist('kstest', 'file') == 2 ...
                && exist('fcdf', 'file') == 2, ...
                'kstest and fcdf (Statistics Toolbox) are required.');

            for spec = {[3, 4], [5, 1], [2, 0.5]}
                d = spec{1}(1);
                nu = spec{1}(2);
                R = localEquiCorrelation(d, 0.3);
                X = tCopulaSample(testCase.NumDraws, d, R, 1 / nu, ...
                    7700 + 10 * d + round(10 * nu));

                statistic = localMahalanobisPerDimension(X, R);
                [rejected, pValue] = kstest(statistic, ...
                    'CDF', [statistic(:), fcdf(statistic(:), d, nu)]);

                label = sprintf('d = %d, nu = %g', d, nu);
                testCase.verifyFalse(rejected, sprintf( ...
                    ['%s: X''R^-1 X / d must follow F(%d, %g); the ' ...
                    'Kolmogorov-Smirnov p-value was %.4g.'], ...
                    label, d, nu, pValue));
                testCase.verifyGreaterThan(pValue, ...
                    testCase.SignificanceLevel, label);
            end
        end

        function gaussianDrawFollowsTheChiSquareLimit(testCase)
            % p = 0 is the Gaussian member, where the same form is chi-square(d)/d, the nu -> Inf limit of F(d, nu).
            % A sampler that quietly applied a mixing variate at p = 0 would fail here.
            testCase.assumeTrue(exist('kstest', 'file') == 2 ...
                && exist('chi2cdf', 'file') == 2, ...
                'kstest and chi2cdf (Statistics Toolbox) are required.');
            d = 4;
            R = localEquiCorrelation(d, 0.3);
            X = tCopulaSample(testCase.NumDraws, d, R, 0, 7801);

            statistic = localMahalanobisPerDimension(X, R);
            [rejected, pValue] = kstest(statistic, ...
                'CDF', [statistic(:), chi2cdf(d * statistic(:), d)]);

            testCase.verifyFalse(rejected, sprintf( ...
                ['at p = 0 the Mahalanobis form must follow chi-square(%d)/%d; ' ...
                'the Kolmogorov-Smirnov p-value was %.4g.'], d, d, pValue));
            testCase.verifyGreaterThan(pValue, testCase.SignificanceLevel);
        end
    end
end


function statistic = localMahalanobisPerDimension(X, R)
    %LOCALMAHALANOBISPERDIMENSION X' R^-1 X / d, row by row, through Cholesky.
    d = size(X, 2);
    L = chol(R);                       % R = L' * L
    statistic = sum((X / L) .^ 2, 2) / d;
end


function R = localEquiCorrelation(d, rho)
    R = (1 - rho) * eye(d) + rho * ones(d);
end
