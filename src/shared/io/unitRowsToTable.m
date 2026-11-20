function T = unitRowsToTable(rows)
    % UNITROWSTOTABLE Flat unit rows as a CSV-ready table.
    %
    % T = unitRowsToTable(rows) turns the study's flat unit rows (as tFitUnit builds them) into one table with a scalar in every cell, which is what writetable can express and what every renderer reads.
    %
    % Two groups of fields do not survive the trip, by design:
    %   - the packed matrices (tau_matrix, R) and the profile vectors (profile_p_grid, profile_loglik_grid) stay in the MAT files;
    %   - channels (1 x 20, NaN padded) becomes channels_str, the space-separated channel list, plus the columns ch01..ch20 that keep the padded numbers.
    % Every other field keeps its name and its position, so the column order of the CSV is the field order of the row.
    %
    % Either layout is accepted.
    % The unit rows live as a struct array while a phase is building them and as the column struct of unitRowsToColumns once they are on disk, and both produce the identical table: a scalar struct carrying a <name>_labels list is the column layout, its coded text columns are decoded through those lists, and its dataset_key (stored once) is repeated down the column.
    %
    % Input:
    %   rows - 1 x n struct array of unit rows, all with the same field set, or the 1 x 1 column struct unitRowsToColumns made from one.
    % Output:
    %   T    - n-row table.

    arguments
        rows (1, :) struct
    end

    if isscalar(rows) && any(endsWith(fieldnames(rows), '_labels'))
        T = localColumnTable(rows);
        return
    end

    assert(~isempty(rows), 'unitRowsToTable:NoRows', ...
           'A unit table needs at least one row; got an empty struct array.');

    packed = localPackedFields();
    names = reshape(fieldnames(rows), 1, []);
    columns = {};
    column_names = {};
    for k = 1:numel(names)
        name = names{k};
        if ismember(name, packed)
            continue
        end
        if strcmp(name, 'channels')
            [channels_str, channel_columns] = localChannelColumns(rows);
            columns{end + 1} = channels_str; %#ok<AGROW>
            column_names{end + 1} = 'channels_str'; %#ok<AGROW>
            for c = 1:size(channel_columns, 2)
                columns{end + 1} = channel_columns(:, c); %#ok<AGROW>
                column_names{end + 1} = sprintf('ch%02d', c); %#ok<AGROW>
            end
            continue
        end
        columns{end + 1} = localScalarColumn(rows, name); %#ok<AGROW>
        column_names{end + 1} = name; %#ok<AGROW>
    end

    T = table(columns{:}, 'VariableNames', column_names);
end

function packed = localPackedFields()
    % The fields that stay in the MAT: they are matrices, not table cells.
    packed = {'tau_matrix', 'R', 'profile_p_grid', 'profile_loglik_grid'};
end

function T = localColumnTable(units)
    % The same table, built straight from a column struct.
    % The rules are the row path's, applied one column at a time: the packed matrices and the profiles are dropped, channels becomes channels_str plus ch01..ch20, a coded text column is decoded through its label list, and dataset_key (stored once) is repeated down the column.
    packed = localPackedFields();
    names = reshape(fieldnames(units), 1, []);
    per_unit = names(~endsWith(names, '_labels') & ~strcmp(names, 'dataset_key'));
    assert(~isempty(per_unit), 'unitRowsToTable:NoUnitColumn', ...
           'The column struct carries no per-unit column; its fields are %s.', ...
           strjoin(names, ', '));
    n = size(units.(per_unit{1}), 1);
    assert(n > 0, 'unitRowsToTable:NoRows', ...
           'A unit table needs at least one row; the column struct holds 0 unit(s).');

    columns = {};
    column_names = {};
    for k = 1:numel(names)
        name = names{k};
        if ismember(name, packed) || endsWith(name, '_labels')
            continue
        end
        if strcmp(name, 'dataset_key')
            columns{end + 1} = repmat({char(units.dataset_key)}, n, 1); %#ok<AGROW>
            column_names{end + 1} = name; %#ok<AGROW>
            continue
        end
        if strcmp(name, 'channels')
            [channels_str, channel_columns] = localChannelColumnsFromMatrix(units.channels);
            columns{end + 1} = channels_str; %#ok<AGROW>
            column_names{end + 1} = 'channels_str'; %#ok<AGROW>
            for c = 1:size(channel_columns, 2)
                columns{end + 1} = channel_columns(:, c); %#ok<AGROW>
                column_names{end + 1} = sprintf('ch%02d', c); %#ok<AGROW>
            end
            continue
        end
        if isfield(units, [name '_labels'])
            labels = reshape(units.([name '_labels']), 1, []);
            columns{end + 1} = reshape(labels(double(units.(name))), [], 1); %#ok<AGROW>
            column_names{end + 1} = name; %#ok<AGROW>
            continue
        end
        value = units.(name);
        assert(size(value, 2) == 1, 'unitRowsToTable:UnsupportedField', ...
               ['Column "%s" is %s. A unit table holds one value per unit, so a ' ...
                'wider column belongs in the MAT beside the packed matrices.'], ...
               name, mat2str(size(value)));
        columns{end + 1} = value; %#ok<AGROW>
        column_names{end + 1} = name; %#ok<AGROW>
    end

    T = table(columns{:}, 'VariableNames', column_names);
end

function [channels_str, channel_columns] = localChannelColumnsFromMatrix(channels)
    % The [n x 20] padded channel matrix as the readable string and its own columns.
    channel_columns = channels;
    channels_str = cell(size(channels, 1), 1);
    for k = 1:size(channels, 1)
        present = channels(k, isfinite(channels(k, :)));
        channels_str{k} = strtrim(sprintf('%d ', present));
    end
end

function [channels_str, channel_columns] = localChannelColumns(rows)
    % The padded channel list as both a readable string and its numeric columns.
    width = numel(rows(1).channels);
    assert(all(arrayfun(@(r) numel(r.channels) == width, rows)), ...
           'unitRowsToTable:RaggedChannels', ...
           ['Every unit row must carry the same channel width; row 1 has %d ' ...
            'entries and another row does not.'], width);
    channel_columns = reshape([rows.channels], width, []).';
    channels_str = cell(numel(rows), 1);
    for k = 1:numel(rows)
        present = rows(k).channels(isfinite(rows(k).channels));
        channels_str{k} = strtrim(sprintf('%d ', present));
    end
end

function column = localScalarColumn(rows, name)
    % One field as a table column: text becomes a cellstr, everything else a vector.
    value = rows(1).(name);
    if ischar(value) || isstring(value)
        column = cellstr(reshape(string({rows.(name)}), [], 1));
    elseif (isnumeric(value) || islogical(value)) && isscalar(value)
        column = reshape([rows.(name)], [], 1);
    else
        error('unitRowsToTable:UnsupportedField', ...
              ['Field "%s" is a %s of size %s. A unit table holds scalar ' ...
               'numeric, logical and text fields only; a matrix or vector ' ...
               'field belongs in the MAT file.'], ...
              name, class(value), mat2str(size(value)));
    end
end
