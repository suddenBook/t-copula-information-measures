function tests = computePairEmpiricalCopulaDensityTest
    % COMPUTEPAIREMPIRICALCOPULADENSITYTEST The empirical panel is a density, not a PMF.
    %
    % The three Phase 5 panels are compared cell by cell, so they must be on one scale: each is a cell-averaged density (cell probability / cell area) that integrates to 1 over the unit square.
    % These checks pin that scale, and the independence case pins the value it takes when the copula is uniform.
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    % Put src/ on the path and share one 20 x 20 grid between the tests.
    here = fileparts(mfilename('fullpath'));
    repo = fileparts(here);
    addpath(genpath(fullfile(repo, 'src')));
    testCase.TestData.edges = 0:0.05:1;
end

function testDensityIntegratesToOne(testCase)
    % sum(density .* cell_area) = 1 on any grid, uniform or not.
    edges = testCase.TestData.edges;
    stream = RandStream('twister', 'Seed', 20260922);
    x = randn(stream, 4000, 1);
    y = 0.7 * x + sqrt(1 - 0.49) * randn(stream, 4000, 1);

    [density, pmf] = computePairEmpiricalCopulaDensity(x, y, edges, edges);
    area = diff(edges).' * diff(edges);
    verifyEqual(testCase, sum(density .* area, 'all'), 1, 'AbsTol', 1e-12);
    verifyEqual(testCase, sum(pmf, 'all'), 1, 'AbsTol', 1e-12);
    verifyEqual(testCase, density, pmf ./ area, 'AbsTol', 1e-12);

    % A ragged grid exercises the per-cell area rather than one shared scalar.
    ragged = [0 0.1 0.15 0.4 0.75 0.9 1];
    [ragged_density, ragged_pmf] = computePairEmpiricalCopulaDensity( ...
                                                                     x, y, ragged, edges);
    ragged_area = diff(ragged).' * diff(edges);
    verifyEqual(testCase, sum(ragged_density .* ragged_area, 'all'), 1, ...
                'AbsTol', 1e-12);
    verifyEqual(testCase, sum(ragged_pmf, 'all'), 1, 'AbsTol', 1e-12);
end

function testIndependentDataGivesUnitDensity(testCase)
    % Independent margins have the uniform copula, whose cell-averaged density is 1 everywhere.
    % With 20 x 20 cells and 200,000 draws the sampling spread of a cell is about sqrt(400/200000) = 4.5 percent, so 0.15 is a bound of about 3.3 spreads on the largest of the 400 cell departures.
    edges = testCase.TestData.edges;
    stream = RandStream('twister', 'Seed', 424242);
    n = 200000;
    density = computePairEmpiricalCopulaDensity( ...
                                                randn(stream, n, 1), randn(stream, n, 1), edges, edges);

    verifyEqual(testCase, mean(density, 'all'), 1, 'AbsTol', 1e-10);
    verifyLessThan(testCase, max(abs(density - 1), [], 'all'), 0.15);
end

function testGridMustSpanTheUnitSquare(testCase)
    % A truncated grid would silently renormalise the histogram.
    verifyError(testCase, @() computePairEmpiricalCopulaDensity( ...
                                                                (1:100).', (100:-1:1).', 0:0.1:0.9, 0:0.1:1), ...
                'computePairEmpiricalCopulaDensity:IncompleteGrid');
end
