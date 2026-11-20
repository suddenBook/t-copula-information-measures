function columns = unitRowsToColumns(rows)
    % UNITROWSTOCOLUMNS Flat unit rows as one struct whose fields are per-unit columns.
    %
    % columns = unitRowsToColumns(rows) turns a struct array of unit rows (the field set tFitUnit produces) into a single struct with one array per field and one row of that array per unit.
    % It exists because of how a -v7.3 MAT file stores a struct array: one HDF5 dataset per field per element, about 400 bytes of header each.
    % With about 180,000 unit rows per recording (10,000 channel sets at each dimension from 3 to 20, plus the pairs), that is gigabytes of headers for tens of megabytes of numbers, and loading such a file takes minutes.
    % As columns, the same numbers are one dataset per field, a file about the size of the numbers, read in one go.
    %
    % The layout:
    %   scalar numeric and logical fields  ->  [n x 1] double (logicals as 0/1)
    %   channels (1 x 20, NaN padded)      ->  [n x 20] double
    %   profile_p_grid, profile_loglik_grid->  [n x 41] double at the default profile_grid_size
    %   status, reason and any other text  ->  [n x 1] int8 codes, with the sorted cellstr <name>_labels beside them
    %   dataset_key                        ->  the char, stored once
    % The field order of the row is preserved, and each <name>_labels sits directly after the field it decodes.
    %
    % The numeric class is not preserved, by design: every numeric column is stored as double, and a logical column comes back from unitColumnsToRows as double 0/1.
    % isequal treats the two as equal, every reader in this project tests them with logical(), and writetable writes both as 0/1, so the CSV is the same either way.
    %
    % Input:
    %   rows    - 1 x n struct array of unit rows, all with the same field set.
    %             An empty struct array is allowed and gives empty columns, so that a dimension with no channel set still round-trips.
    % Output:
    %   columns - 1 x 1 struct; unitColumnsToRows inverts it, with every numeric and logical field returned as double.
    %
    % See also unitColumnsToRows, unitRowsToTable.

    arguments
        rows (1, :) struct
    end

    names = reshape(fieldnames(rows), 1, []);
    n = numel(rows);
    columns = struct();
    for k = 1:numel(names)
        name = names{k};
        if strcmp(name, 'dataset_key')
            columns.dataset_key = localSharedKey(rows);
            continue
        end
        if n == 0
            % With no row there is no value, no width and nothing that says which fields carried text.
            % The empty column keeps the field set, which is all a 1 x 0 struct array can carry, and no label list is written.
            columns.(name) = zeros(0, 1);
            continue
        end
        if ischar(rows(1).(name)) || isstring(rows(1).(name))
            [codes, labels] = localEncodeText(rows, name);
            columns.(name) = codes;
            columns.([name '_labels']) = labels;
            continue
        end
        columns.(name) = localStackNumeric(rows, name);
    end
end

function key = localSharedKey(rows)
    % dataset_key is stored once, because every unit row of one file is one recording's.
    if isempty(rows)
        key = '';
        return
    end
    distinct = unique(cellstr(string({rows.dataset_key})));
    assert(isscalar(distinct), 'unitRowsToColumns:MixedDatasetKey', ...
           ['A column struct stores dataset_key once, so every unit row must carry ' ...
            'the same one; %d rows carry %d distinct keys (%s).'], ...
           numel(rows), numel(distinct), strjoin(reshape(distinct, 1, []), ', '));
    key = char(rows(1).dataset_key);
end

function [codes, labels] = localEncodeText(rows, name)
    % A text field as int8 codes into the sorted list of its distinct values.
    values = cellstr(string({rows.(name)}));
    labels = reshape(unique(values), 1, []);
    assert(numel(labels) <= double(intmax('int8')), 'unitRowsToColumns:TooManyLabels', ...
           ['Field "%s" takes %d distinct values over %d unit rows; a coded column ' ...
            'holds at most %d (int8). A text field with one value per unit is not a ' ...
            'label column and does not belong in a MAT.'], ...
           name, numel(labels), numel(rows), double(intmax('int8')));
    [~, code] = ismember(values, labels);
    codes = int8(reshape(code, [], 1));
end

function M = localStackNumeric(rows, name)
    % One numeric or logical field as [n x width]; logicals become double 0/1.
    n = numel(rows);
    if ~(isnumeric(rows(1).(name)) || islogical(rows(1).(name)))
        error('unitRowsToColumns:UnsupportedField', ...
              ['Field "%s" is a %s. A column struct holds numeric, logical and ' ...
               'text fields only; a cell or nested struct field does not belong ' ...
               'in a unit row.'], name, class(rows(1).(name)));
    end
    % The shape is checked, not just the count: a [w x 1] field would stack just as happily and come back from unitColumnsToRows as [1 x w], silently transposed, so a unit row field must be a scalar or a row vector.
    shape = size(rows(1).(name));
    if ~(isscalar(rows(1).(name)) || isrow(rows(1).(name)))
        error('unitRowsToColumns:UnsupportedShape', ...
              ['Field "%s" is %s. A unit row holds scalars and row vectors; a ' ...
               'column vector or a matrix has no unambiguous column form.'], ...
              name, mat2str(shape));
    end
    ragged = find(arrayfun(@(r) ~isequal(size(r.(name)), shape), rows), 1);
    if ~isempty(ragged)
        error('unitRowsToColumns:RaggedField', ...
              ['Field "%s" must have the same shape in every unit row: ' ...
               'row 1 holds %s and row %d holds %s.'], name, mat2str(shape), ...
              ragged, mat2str(size(rows(ragged).(name))));
    end
    width = shape(2);
    M = reshape(double([rows.(name)]), width, n).';
end
