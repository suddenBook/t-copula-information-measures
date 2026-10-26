%LOADPROCESSEDDATA Load one data matrix from a MAT file and return its size.
%
% [X, n_samples, dimension] = loadProcessedData(dataPath, varName) loads the variable varName from the MAT file dataPath and returns it with its row count n_samples and column count dimension, which are samples and channels for a [samples x channels] matrix.
% A file that does not contain varName is an error.

function [X, n_samples, dimension] = loadProcessedData(dataPath, varName)

    load(dataPath, varName);

    if ~exist(varName, 'var')
        error('loadProcessedData: %s not found in %s', varName, dataPath);
    end

    X = eval(varName);

    n_samples = size(X, 1);
    dimension = size(X, 2);
    fprintf('Loaded %s: %d samples, %d dimensions\n', dataPath, n_samples, dimension);
end
