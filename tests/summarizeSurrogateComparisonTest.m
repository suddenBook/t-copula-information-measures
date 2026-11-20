classdef summarizeSurrogateComparisonTest < matlab.unittest.TestCase
    % SUMMARIZESURROGATECOMPARISONTEST Tests of the per-dimension surrogate comparison of Phase 3.
    %
    % One recording's observed shape estimates are compared with its two surrogate realizations and reduced to a table per dimension and a floor table.
    % The suite checks that the seeded observed halves are deterministic, balanced and keyed on the unit rather than on the row; that a failed fit enters no statistic and is counted; that the overlap columns are distributionOverlap on the successful fits; that the floors are per dimension and realization, with mutual information at d = 2 only; and that the column names are the ones the renderers and Phase 4 read.
    % The suite uses synthetic inputs only and reads no recordings.
    % Every sample comes from a private stream, so the global generator is untouched.

    methods (TestClassSetup)

        function addSourceTree(testCase)
            sourceTree = fullfile( ...
                                  fileparts(fileparts(mfilename('fullpath'))), 'src');
            assert(isfolder(sourceTree), ...
                   'summarizeSurrogateComparisonTest:SourceTreeNotFound', ...
                   'expected this file under <repo>/tests/; got %s', ...
                   mfilename('fullpath'));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                                                                       sourceTree, IncludingSubfolders = true));
        end

    end

    methods (Test)

        function halvesAreDeterministicAndBalanced(testCase)
            [observed, surrogate, opts] = localFixture();

            [~, ~, half] = summarizeSurrogateComparison(observed, surrogate, 7, opts);
            [~, ~, again] = summarizeSurrogateComparison(observed, surrogate, 7, opts);

            testCase.verifyEqual(again, half, 'the split must be reproducible.');
            for d = [2, 3]
                at_d = observed.d == d;
                testCase.verifyTrue(all(ismember(half(at_d), [1, 2])));
                testCase.verifyLessThanOrEqual( ...
                                               abs(nnz(half(at_d) == 1) - nnz(half(at_d) == 2)), 1, ...
                                               sprintf('the halves at d = %d must differ by at most one.', d));
            end

            % The documented rule, rebuilt by hand at d = 3: the channel sets in unit_index order, split by randperm on the offset group stream.
            at_d = find(observed.d == 3);
            [~, order] = sort(observed.unit_index(at_d));
            at_d = at_d(order);
            stream = RandStream('twister', 'Seed', ...
                                channelGroupSeed(7, 3) + opts.halves_seed_offset);
            permutation = randperm(stream, numel(at_d));
            expected = 2 * ones(numel(at_d), 1);
            expected(permutation(1:ceil(numel(at_d) / 2))) = 1;
            testCase.verifyEqual(half(at_d), expected);

            % The split follows the channel set, not the row it arrives in.
            shuffle = randperm(RandStream('twister', 'Seed', 99), numel(observed.d));
            [~, ~, shuffled] = summarizeSurrogateComparison( ...
                                                            localRows(observed, shuffle), localRows(surrogate, shuffle), 7, opts);
            testCase.verifyEqual(shuffled, half(shuffle));

            % Another recording draws another split.
            [~, ~, other] = summarizeSurrogateComparison(observed, surrogate, 8, opts);
            testCase.verifyNotEqual(other, half);
        end

        function tinyDimensionsAreSummarized(testCase)
            % One, two or three channel sets at a dimension (a recording with barely d channels, or a small pilot) leave a half with one value or none, and the tables must still be produced.
            [observed, surrogate, opts] = localFixture();
            triples = find(observed.d == 3);
            pairs = find(observed.d == 2);
            for keep = 1:3
                rows = [triples(1:keep); pairs(1:keep)];
                [comparison, floors] = summarizeSurrogateComparison( ...
                                                                    localRows(observed, rows), localRows(surrogate, rows), 7, opts);
                testCase.verifyEqual(sort(comparison.d), [2; 3]);
                testCase.verifyEqual(comparison.n_units(comparison.d == 3), keep);
                overlap = comparison.ovl_obs_sur;
                testCase.verifyTrue(all(isnan(overlap) | (overlap >= 0 & overlap <= 1)));
                testCase.verifyNotEmpty(floors);
            end
        end

        function failedFitsAreExcludedAndCounted(testCase)
            [observed, surrogate, opts] = localFixture();
            % Failed fits that keep a finite, far-off estimate would move every overlap and quantile at d = 3 if they were counted.
            observed.numerical_ok(observed.d == 3 & observed.unit_index <= 3) = false;
            observed.p_hat(observed.d == 3 & observed.unit_index <= 3) = 2.9;
            failed = find(surrogate.p_hat(:, 1) > 0 & observed.d == 3, 2);
            surrogate.numerical_ok(failed, 1) = false;
            surrogate.p_hat(failed, 1) = 2.9;
            surrogate.status(failed, 1) = localCode(surrogate, 'numerical_failure');

            comparison = summarizeSurrogateComparison(observed, surrogate, 7, opts);

            row = comparison(comparison.d == 3, :);
            at_d = observed.d == 3;
            ok_obs = at_d & logical(observed.numerical_ok);
            ok_sur = at_d & logical(surrogate.numerical_ok(:, 1));
            testCase.verifyEqual(row.n_units, nnz(at_d));
            testCase.verifyEqual(row.n_obs_ok, nnz(at_d) - 3);
            testCase.verifyEqual(row.n_sur1_ok, nnz(at_d) - 2);
            testCase.verifyEqual(row.n_sur2_ok, nnz(at_d));
            testCase.verifyEqual(row.sur_status_numerical_failure, 2);
            testCase.verifyEqual(row.sur_status_ok + row.sur_status_heavy_boundary ...
                                 + row.sur_status_ambiguous_profile + row.sur_status_numerical_failure ...
                                 + row.sur_status_degenerate_pair, nnz(at_d), ...
                                 'every realization fit is counted under exactly one status.');
            testCase.verifyEqual(row.ovl_obs_sur, distributionOverlap( ...
                                                                      observed.p_hat(ok_obs), surrogate.p_hat(ok_sur, 1), ...
                                                                      opts.overlap_bins, opts.atoms), AbsTol = 1e-12);
            testCase.verifyEqual(row.mass_outside, 1 - row.ovl_obs_sur, AbsTol = 1e-12);
            testCase.verifyEqual(row.obs_q50, median(observed.p_hat(ok_obs)), ...
                                 AbsTol = 1e-12);
            testCase.verifyEqual(row.sur_fraction_p0, ...
                                 mean(surrogate.p_hat(ok_sur, 1) == 0), AbsTol = 1e-12);
            testCase.verifyLessThan(row.obs_q975, 2.9, ...
                                    'a failed fit''s estimate must not reach the observed quantiles.');
        end

        function yardsticksAreTheirOwnOverlaps(testCase)
            [observed, surrogate, opts] = localFixture();

            [comparison, ~, half] = summarizeSurrogateComparison( ...
                                                                 observed, surrogate, 7, opts);

            at_d = observed.d == 2;
            row = comparison(comparison.d == 2, :);
            testCase.verifyEqual(row.ovl_sur_sur2, distributionOverlap( ...
                                                                       surrogate.p_hat(at_d, 1), surrogate.p_hat(at_d, 2), ...
                                                                       opts.overlap_bins, opts.atoms), AbsTol = 1e-12);
            testCase.verifyEqual(row.ovl_obs_halves, distributionOverlap( ...
                                                                         observed.p_hat(at_d & half == 1), observed.p_hat(at_d & half == 2), ...
                                                                         opts.overlap_bins, opts.atoms), AbsTol = 1e-12);

            % A surrogate that reproduces the observed fits overlaps them fully.
            surrogate.p_hat(:, 1) = observed.p_hat;
            comparison = summarizeSurrogateComparison(observed, surrogate, 7, opts);
            testCase.verifyEqual(comparison.ovl_obs_sur, [1; 1], AbsTol = 1e-12);
            testCase.verifyEqual(comparison.mass_outside, [0; 0], AbsTol = 1e-12);
        end

        function floorsArePerDimensionAndRealization(testCase)
            [observed, surrogate, opts] = localFixture();
            surrogate.numerical_ok(observed.d == 3, 2) = false;

            [~, floors] = summarizeSurrogateComparison(observed, surrogate, 7, opts);

            testCase.verifyEqual(floors.Properties.VariableNames, ...
                                 {'d', 'realization', 'measure', 'n', 'q50', 'q90', 'q95', 'q99'});
            testCase.verifyEqual(sort(floors.measure(floors.d == 2 & ...
                                                     floors.realization == 1)), {'DTC'; 'MI'; 'O'; 'TC'});
            testCase.verifyEqual(sort(floors.measure(floors.d == 3 & ...
                                                     floors.realization == 1)), {'DTC'; 'O'; 'TC'}, ...
                                 'mutual information is a pair measure and has no floor above d = 2.');
            testCase.verifyEqual(unique(floors.realization), [1; 2]);

            at_d = observed.d == 3;
            differences = tInformationDifferences(3, surrogate.p_hat(at_d, 1));
            row = floors(floors.d == 3 & floors.realization == 1 & ...
                         strcmp(floors.measure, 'TC'), :);
            testCase.verifyEqual(row.n, nnz(at_d));
            testCase.verifyEqual(row.q95, prctile(differences.TC, 95), AbsTol = 1e-12);
            testCase.verifyGreaterThanOrEqual(row.q99, row.q95);

            % A realization with no successful fit keeps its rows, empty.
            empty = floors(floors.d == 3 & floors.realization == 2, :);
            testCase.verifyEqual(empty.n, [0; 0; 0]);
            testCase.verifyTrue(all(isnan(empty.q95)));
        end

        function columnNamesAreTheDeclaredOnes(testCase)
            [observed, surrogate, opts] = localFixture();

            comparison = summarizeSurrogateComparison(observed, surrogate, 7, opts);

            levels = {'q025', 'q05', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'q975'};
            statuses = {'ok', 'heavy_boundary', 'ambiguous_profile', ...
                        'numerical_failure', 'degenerate_pair'};
            expected = [{'d', 'n_units', 'n_obs_ok', 'n_sur1_ok', 'n_sur2_ok', ...
                         'ovl_obs_sur', 'mass_outside', 'ovl_sur_sur2', 'ovl_obs_halves'}, ...
                        strcat('obs_', levels), strcat('sur_', levels), strcat('sur2_', levels), ...
                        {'obs_fraction_p0', 'sur_fraction_p0', 'sur2_fraction_p0', ...
                         'obs_fraction_heavy', 'sur_fraction_heavy', 'sur2_fraction_heavy'}, ...
                        strcat('sur_status_', statuses), strcat('sur2_status_', statuses)];
            testCase.verifyEqual(comparison.Properties.VariableNames, expected);
            testCase.verifyEqual(comparison.d, [2; 3]);
        end

        function aThirdRealizationIsRefused(testCase)
            [observed, surrogate, opts] = localFixture();
            surrogate.p_hat(:, 3) = 0;
            testCase.verifyError( ...
                                 @() summarizeSurrogateComparison(observed, surrogate, 7, opts), ...
                                 'summarizeSurrogateComparison:RealizationShapeMismatch');
        end

    end
end

function [observed, surrogate, opts] = localFixture()
    % Nine pairs and twelve triples, observed and on two realizations, shaped like p_hat: about half exactly Gaussian and a positive body, the observed body heavier than the surrogate one.
    % The rows arrive with the triples first and out of unit_index order, since nothing guarantees an order.
    stream = RandStream('twister', 'Seed', 11);
    d = [3 * ones(12, 1); 2 * ones(9, 1)];
    unit_index = [randperm(stream, 12).'; randperm(stream, 9).'];
    m = numel(d);
    observed = struct('d', d, 'unit_index', unit_index, ...
                      'p_hat', localShape(stream, m, 0.3), 'numerical_ok', true(m, 1), ...
                      'heavy_boundary', false(m, 1));
    labels = {'ambiguous_profile', 'degenerate_pair', 'heavy_boundary', ...
              'numerical_failure', 'ok'};
    surrogate = struct('p_hat', [localShape(stream, m, 0.05), ...
                                 localShape(stream, m, 0.05)], 'numerical_ok', true(m, 2), ...
                       'heavy_boundary', false(m, 2), 'status', int8(5 * ones(m, 2)), ...
                       'status_labels', {labels});
    opts = struct('overlap_bins', 4, 'atoms', [0, 3], ...
                  'quantile_levels', [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5], ...
                  'halves_seed_offset', 2);
end

function p = localShape(stream, m, scale)
    % m values shaped like p_hat: an exponential body of the given scale, with about half set exactly to 0.
    p = -log(rand(stream, m, 1)) * scale;
    p(rand(stream, m, 1) < 0.5) = 0;
end

function code = localCode(surrogate, label)
    % The status code of one label in the fixture's status_labels.
    code = int8(find(strcmp(surrogate.status_labels, label)));
end

function columns = localRows(columns, order)
    % The same columns with their rows permuted; the label list is not a column.
    for name = reshape(fieldnames(columns), 1, [])
        if ~strcmp(name{1}, 'status_labels')
            columns.(name{1}) = columns.(name{1})(order, :);
        end
    end
end
