function writeUnitRowsCsv(rows, path)
    %WRITEUNITROWSCSV Write flat unit rows to a CSV, atomically.
    %
    % writeUnitRowsCsv(rows, path) is unitRowsToTable followed by writeTableAtomic: the one place a phase turns its unit rows into a CSV the renderers read, so the pairs.csv and groups_long.csv of Phase 2 carry the same columns in the same order.
    %
    % Inputs:
    %   rows - 1 x n struct array of unit rows, with the field set tFitUnit produces.
    %   path - absolute path of the CSV to write.

    arguments
        rows (1, :) struct
        path (1, :) char
    end

    writeTableAtomic(unitRowsToTable(rows), path);
end
