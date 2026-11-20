function rows = unitColumnsToRows(columns, idx)
    % UNITCOLUMNSTOROWS Inverse of unitRowsToColumns, with every numeric and logical field returned as double.
    %
    % rows = unitColumnsToRows(columns) rebuilds the struct array of unit rows a column struct was made from: one element per unit, every field back in its own shape and its own place, the coded text columns decoded through their label lists and dataset_key copied into every row.
    %
    % rows = unitColumnsToRows(columns, idx) rebuilds only the units idx names, in the order given, which is how a phase takes the few rows it wants out of a recording's whole unit table without materialising the rest.
    %
    % A column struct is the storage form (see unitRowsToColumns); a row struct is the convenient form wherever per-unit fields are added, sorted or passed one at a time.
    % Prefer indexing the columns directly (columns.p_hat(mask)) and come here only when a row struct is genuinely simpler.
    %
    % The numeric class is not restored: a field the rows carried as logical, single or an integer type comes back as double (a logical as 0/1), because that is how the column struct stores it.
    % isequal treats the two as equal, so a round trip still compares equal (with isequaln where the rows carry NaN).
    %
    % Inputs:
    %   columns - 1 x 1 struct written by unitRowsToColumns.
    %   idx     - (optional) unit indices to rebuild, in any shape; omitted means every unit, and an empty idx means none.
    %             A logical mask is refused, because idx is an index list; pass find(mask).
    % Output:
    %   rows    - 1 x numel(idx) struct array of unit rows; a 1 x 0 empty struct with the same field set when idx selects nothing.
    %
    % See also unitRowsToColumns, unitRowsToTable.

    arguments
        columns (1, 1) struct
        idx {mustBeInteger, mustBePositive} = zeros(1, 0)
    end

    if islogical(idx)
        % idx lists unit indices, so a logical mask is refused rather than guessed at.
        error('unitColumnsToRows:LogicalIndex', ...
              ['idx is a list of unit indices, not a logical mask over %d unit(s); ' ...
               'pass find(mask).'], size(columns.(localFirstUnitColumn(columns)), 1));
    end

    names = reshape(fieldnames(columns), 1, []);
    data_names = names(~endsWith(names, '_labels'));
    n = localUnitCount(columns);
    if nargin < 2
        idx = 1:n;
    end
    idx = reshape(idx, 1, []);
    out_of_range = find(idx > n, 1);
    if ~isempty(out_of_range)
        error('unitColumnsToRows:IndexOutOfRange', ...
              'Unit index %d was asked for, but the column struct holds %d unit(s).', ...
              idx(out_of_range), n);
    end

    m = numel(idx);
    pairs = cell(1, 2 * numel(data_names));
    for k = 1:numel(data_names)
        name = data_names{k};
        pairs{2 * k - 1} = name;
        pairs{2 * k} = localFieldValues(columns, name, idx, m);
    end
    rows = struct(pairs{:});
end

function n = localUnitCount(columns)
    % The unit count is the row count of the first column that carries one.
    n = size(columns.(localFirstUnitColumn(columns)), 1);
end

function name = localFirstUnitColumn(columns)
    % The first field that holds one row per unit, which sets the unit count.
    names = reshape(fieldnames(columns), 1, []);
    counted = names(~endsWith(names, '_labels') & ~strcmp(names, 'dataset_key'));
    assert(~isempty(counted), 'unitColumnsToRows:NoUnitColumn', ...
           ['A column struct needs at least one per-unit column besides ' ...
            'dataset_key; this one carries only %s.'], strjoin(names, ', '));
    name = counted{1};
end

function values = localFieldValues(columns, name, idx, m)
    % One field as the 1 x m cell struct() needs to build the struct array in one call.
    if strcmp(name, 'dataset_key')
        values = repmat({char(columns.dataset_key)}, 1, m);
        return
    end
    label_field = [name '_labels'];
    if isfield(columns, label_field)
        labels = reshape(columns.(label_field), 1, []);
        code = double(reshape(columns.(name)(idx), 1, []));
        bad = find(code < 1 | code > numel(labels), 1);
        if ~isempty(bad)
            error('unitColumnsToRows:LabelCodeOutOfRange', ...
                  ['Unit %d carries the code %d for field "%s", which lists %d ' ...
                   'label(s); the codes and %s disagree.'], ...
                  idx(bad), code(bad), name, numel(labels), label_field);
        end
        values = labels(code);
        return
    end
    block = columns.(name);
    values = reshape(num2cell(block(idx, :), 2), 1, m);
end
