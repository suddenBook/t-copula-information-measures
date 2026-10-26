%VALIDATEDATA Check that loaded data form a numeric 2-D matrix without NaN or Inf.
%
% Raises an assertion error on the first check that fails.
%
% Inputs:
%   data - the loaded data matrix.

function validateData(data)

    assert(ismatrix(data), 'Data must be a 2D matrix');
    assert(isnumeric(data), 'Data must be numeric');
    assert(~any(isnan(data(:))), 'Data contains NaN values');
    assert(~any(isinf(data(:))), 'Data contains Inf values');

end
