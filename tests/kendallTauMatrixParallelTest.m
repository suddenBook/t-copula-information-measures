classdef kendallTauMatrixParallelTest < matlab.unittest.TestCase
    %KENDALLTAUMATRIXPARALLELTEST The parallel Kendall matrix is the serial one.
    %
    % Phase 3 computes one full-recording Kendall matrix per surrogate realization through kendallTauMatrixParallel and then reads its submatrices, while Phase 2 fits each channel set from kendallTauMatrix.
    % If the two disagreed, the observed fits and their surrogate reference would come from different correlation matrices and the comparison would be meaningless.
    % The tests pin the two to bit equality, including on degenerate inputs where tau-b is not defined, and check that the worker limit is a scheduling choice and that a run asking for zero workers runs serially.
    %
    % The suite uses synthetic inputs only and reads no recordings.

    methods (TestClassSetup)
        function addSourceTree(testCase)
            sourceTree = fullfile( ...
                fileparts(fileparts(mfilename('fullpath'))), 'src');
            assert(isfolder(sourceTree), ...
                'kendallTauMatrixParallelTest:SourceTreeNotFound', ...
                'expected this file under <repo>/tests/; got %s', ...
                mfilename('fullpath'));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                sourceTree, IncludingSubfolders=true));
        end
    end

    methods (Test)
        function equalsTheSerialMatrix(testCase)
            % Same rows, same pairs, same numbers, with AbsTol 0 because both entry points make the same kendallTauKnight call.
            X = localDependentColumns(400, 7, 31);

            testCase.verifyEqual(kendallTauMatrixParallel(X), ...
                kendallTauMatrix(X), ...
                ['the parallel and serial Kendall matrices must be bit ' ...
                'identical: Phase 2 fits from one and Phase 3 judges it ' ...
                'against the other.'], AbsTol=0);
        end

        function equalsTheSerialMatrixWithTies(testCase)
            % Ties change tau-b, so both routes must treat them the same way.
            % Rounding a continuous matrix makes a tied one.
            X = round(localDependentColumns(300, 5, 7) * 3);

            testCase.verifyEqual(kendallTauMatrixParallel(X), ...
                kendallTauMatrix(X), AbsTol=0);
        end

        function narrowestMatrixIsOnePair(testCase)
            % nchoosek(1:2, 2) is the one-row edge case of the pair list.
            X = localDependentColumns(200, 2, 3);
            tau = kendallTauMatrixParallel(X);

            testCase.verifySize(tau, [2, 2]);
            testCase.verifyEqual(diag(tau), ones(2, 1), AbsTol=0);
            testCase.verifyEqual(tau, kendallTauMatrix(X), AbsTol=0);
        end

        function constantColumnPropagatesNaNAsInTheSerialRoute(testCase)
            % A constant channel has no tau-b.
            % Both routes must report the same NaN pattern, which unitCorrelationFromTau reads as 'nonfinite_kendall_tau' rather than inventing a correlation.
            X = localDependentColumns(150, 4, 11);
            X(:, 3) = 2.5;

            tau = kendallTauMatrixParallel(X);

            testCase.verifyEqual(isnan(tau), isnan(kendallTauMatrix(X)));
            testCase.verifyTrue(all(isnan(tau([1, 2, 4], 3))));
            testCase.verifyEqual(tau, kendallTauMatrix(X), AbsTol=0);
        end

        function tooFewChannelsIsNamed(testCase)
            testCase.verifyError(@() kendallTauMatrixParallel(randn(20, 1)), ...
                'kendallTauMatrixParallel:TooFewChannels');
        end

        function theWorkerLimitDoesNotMoveTheMatrix(testCase)
            % The limit is a scheduling choice and nothing else: the serial request must return the same bits as the unrestricted loop and as the serial entry point.
            X = localDependentColumns(250, 6, 23);

            testCase.verifyEqual(kendallTauMatrixParallel(X, 0), ...
                kendallTauMatrixParallel(X), AbsTol=0);
            testCase.verifyEqual(kendallTauMatrixParallel(X, 0), ...
                kendallTauMatrix(X), AbsTol=0);
        end

        function aZeroWorkerRequestOpensNoPool(testCase)
            % TC_WORKERS=0 has to mean serial.
            % An unrestricted parfor uses an open pool, and opens one itself when MATLAB's auto-create preference is on, so a run that asked for no workers could take a whole machine's worth of them while reporting zero.
            % Any pool the tests above opened is closed first, so that no pool afterwards means this call opened none; this test is last so that closing the pool costs the suite nothing.
            X = localDependentColumns(150, 4, 29);
            if license('test', 'Distrib_Computing_Toolbox')
                existing = gcp('nocreate');
                if ~isempty(existing)
                    delete(existing);
                end
            end

            tau = kendallTauMatrixParallel(X, 0);

            testCase.verifyEqual(tau, kendallTauMatrix(X), AbsTol=0);
            if license('test', 'Distrib_Computing_Toolbox')
                testCase.verifyEmpty(gcp('nocreate'), ...
                    'a zero-worker request must not open a parallel pool.');
            end
        end
    end
end


function X = localDependentColumns(n, d, seed)
    % A correlated [samples x channels] matrix, a shared factor plus channel noise, so every pair has a non-trivial tau-b.
    stream = RandStream('twister', 'Seed', seed);
    factor = randn(stream, n, 1);
    X = 0.7 * factor + randn(stream, n, d);
end
