classdef copulaDensityGridTest < matlab.unittest.TestCase
    % COPULADENSITYGRIDTEST Tests of the model density panels of the d = 2 figure.
    %
    % tCopulaPairDensityGrid and gaussianCopulaDensityGrid produce the two model panels that a histogram of pseudo-observations is compared with, and the residual maps and their RMS values are read off that comparison.
    % So the panels have to be cell averages of the copula density, not pointwise evaluations at the cell centres, and they have to be the real copula.
    % The tests check the total mass, the uniform margins, the cell average against the density kernel, the delegation of p = 0 to the Gaussian member rather than to a large-nu approximation, the heavier corners of the t copula, and the refusal of malformed inputs.

    properties (Constant)
        Edges = 0:0.05:1
        Rho = 0.5
    end

    methods (TestClassSetup)

        function addSourceTree(testCase)
            sourceTree = fullfile( ...
                                  fileparts(fileparts(mfilename('fullpath'))), 'src');
            assert(isfolder(sourceTree), ...
                   'copulaDensityGridTest:SourceTreeNotFound', ...
                   'expected this file under <repo>/tests/; got %s', ...
                   mfilename('fullpath'));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                                                                       sourceTree, IncludingSubfolders = true));
        end

    end

    methods (Test)

        function gridIsAProbabilityDistributionWithUniformMargins(testCase)
            % A copula's margins are uniform, so the mass of a row of cells must be the width of its u interval whatever rho and nu are.
            % That is the one property no rescaling of the panel can fake.
            testCase.assumeTrue(exist('mvtcdf', 'file') == 2, ...
                                'mvtcdf (Statistics Toolbox) is required.');
            edges = testCase.Edges;
            widths = diff(edges);

            for p = [0, 0.25, 2]
                [density, mass] = tCopulaPairDensityGrid( ...
                                                         testCase.Rho, p, edges, edges);
                label = sprintf('p = %g', p);

                testCase.verifySize(mass, [numel(edges) - 1, numel(edges) - 1], ...
                                    label);
                testCase.verifyEqual(sum(mass, 'all'), 1, label, AbsTol = 1e-12);
                testCase.verifyGreaterThanOrEqual(min(mass, [], 'all'), -1e-14, ...
                                                  sprintf('%s: a cell mass must not be negative', label));
                testCase.verifyEqual(sum(mass, 2).', widths, ...
                                     sprintf('%s: the u margin must be uniform', label), ...
                                     AbsTol = 1e-12);
                testCase.verifyEqual(sum(mass, 1), widths, ...
                                     sprintf('%s: the v margin must be uniform', label), ...
                                     AbsTol = 1e-12);
                testCase.verifyEqual(density, mass ./ (widths.' * widths), ...
                                     sprintf('%s: the density is the mass over the cell area', ...
                            label), AbsTol = 0);
            end
        end

        function cellAverageMatchesTheDensityKernelOverTheCell(testCase)
            % The panel must be the average of the copula density over the cell.
            % tCopulaLogDensity is an independent route to the same density, so averaging it on a fine subgrid is an external check on the CDF rectangle rule, in an interior cell where the density is smooth.
            testCase.assumeTrue(exist('mvtcdf', 'file') == 2, ...
                                'mvtcdf (Statistics Toolbox) is required.');
            p = 0.25;
            uCell = [0.40, 0.45];
            vCell = [0.55, 0.60];

            density = tCopulaPairDensityGrid(testCase.Rho, p, uCell, vCell);

            fine = (0.5:199.5) / 200;
            u = uCell(1) + diff(uCell) * fine;
            v = vCell(1) + diff(vCell) * fine;
            [UU, VV] = ndgrid(u, v);
            Z = [localSignedQuantile(UU(:), p), localSignedQuantile(VV(:), p)];
            reference = mean(exp(tCopulaLogDensity(Z, ...
                                                   [1, testCase.Rho; testCase.Rho, 1], p)));

            testCase.verifyEqual(density, reference, sprintf( ...
                                                             ['the cell-averaged density %.8f must match the midpoint ' ...
                                                              'average of tCopulaLogDensity over the same cell (%.8f).'], ...
                                                             density, reference), RelTol = 1e-5);
        end

        function gaussianMemberIsDelegatedNotApproximated(testCase)
            % p = 0 is a member of the family, so the t entry point must return the Gaussian panel itself, bit for bit, and not an mvtcdf call at some large nu.
            edges = testCase.Edges;
            [tDensity, tMass] = tCopulaPairDensityGrid( ...
                                                       testCase.Rho, 0, edges, edges);
            [gDensity, gMass] = gaussianCopulaDensityGrid( ...
                                                          testCase.Rho, edges, edges);

            testCase.verifyEqual(tDensity, gDensity, AbsTol = 0);
            testCase.verifyEqual(tMass, gMass, AbsTol = 0);

            % Independence is the one panel with a closed form everywhere.
            flat = gaussianCopulaDensityGrid(0, edges, edges);
            testCase.verifyEqual(flat, ones(size(flat)), ...
                                 'at rho = 0 the copula density is 1 on the whole square.', ...
                                 AbsTol = 1e-10);
        end

        function heavyTailsPutMoreMassInTheCorners(testCase)
            % The t copula is tail dependent and the Gaussian copula is not, so at the same rho the corner cell must carry more mass under the t.
            % A panel that silently ignored nu would fail here.
            testCase.assumeTrue(exist('mvtcdf', 'file') == 2, ...
                                'mvtcdf (Statistics Toolbox) is required.');
            edges = testCase.Edges;
            [~, heavy] = tCopulaPairDensityGrid(testCase.Rho, 0.5, edges, edges);
            [~, gaussian] = gaussianCopulaDensityGrid(testCase.Rho, edges, edges);

            testCase.verifyGreaterThan(heavy(end, end), gaussian(end, end), ...
                                       'the upper corner cell must be heavier under the t copula.');
            testCase.verifyGreaterThan(heavy(1, 1), gaussian(1, 1), ...
                                       'the lower corner cell must be heavier under the t copula.');
        end

        function malformedGridsAreRefusedWithANamedAssumption(testCase)
            edges = testCase.Edges;
            testCase.verifyError( ...
                                 @() gaussianCopulaDensityGrid(1, edges, edges), ...
                                 'gaussianCopulaDensityGrid:CorrelationOutOfRange');
            testCase.verifyError( ...
                                 @() tCopulaPairDensityGrid(-1, 0.25, edges, edges), ...
                                 'tCopulaPairDensityGrid:CorrelationOutOfRange');
            testCase.verifyError( ...
                                 @() gaussianCopulaDensityGrid(0.5, [0, 0.6, 0.4, 1], edges), ...
                                 'copulaCellMassFromCdf:InvalidEdges');
            testCase.verifyError( ...
                                 @() gaussianCopulaDensityGrid(0.5, [0, 1.2], edges), ...
                                 'copulaCellMassFromCdf:InvalidEdges');
        end

    end
end

function z = localSignedQuantile(u, p)
    z = sign(u - 0.5) .* tTailQuantile(2 * min(u, 1 - u), p);
end
