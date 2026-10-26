classdef atomicFileWritersTest < matlab.unittest.TestCase
    %ATOMICFILEWRITERSTEST The MAT and CSV writers stage through their own file.
    %
    % Phase 3's initial shards can decide to create the shared subset files at the same moment, and a resumed Phase 7 array can launch several finalizers of the same outputs, so two processes really do write one destination at once.
    % A staging file shared by name ('<destination>.tmp.mat' for every writer) turns that collision destructive: each writer deletes, overwrites and renames the other's half-written file, and the loser either corrupts the MAT file or publishes a truncated one.
    %
    % One process cannot reproduce a race, so these tests pin the two properties that make the race survivable and that a single process can see: the staging name is not the shared one, so no writer can touch another's staging file, and repeated publication leaves the destination valid with nothing left over beside it.
    %
    % The suite writes scratch files to a temporary folder only and reads no recordings.

    methods (TestClassSetup)
        function addSourceTree(testCase)
            sourceTree = fullfile( ...
                fileparts(fileparts(mfilename('fullpath'))), 'src');
            assert(isfolder(sourceTree), ...
                'atomicFileWritersTest:SourceTreeNotFound', ...
                'expected this file under <repo>/tests/; got %s', ...
                mfilename('fullpath'));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                sourceTree, IncludingSubfolders=true));
        end
    end

    methods (Test)
        function twoMatWritesLeaveOneFileAndNoLeftovers(testCase)
            folder = localScratchFolder(testCase);
            destination = fullfile(folder, 'checkpoint.mat');

            saveStructAtomic(destination, 'payload', struct('value', 1));
            saveStructAtomic(destination, 'payload', struct('value', 2));

            loaded = load(destination, 'payload');
            testCase.verifyEqual(loaded.payload.value, 2, ...
                'the last writer''s data is what the destination holds.');
            testCase.verifyEqual(localFileNames(folder), {'checkpoint.mat'}, ...
                ['a staging file must not survive the write: a leftover is ' ...
                'what a later run would have to guess about.']);
        end

        function twoCsvWritesLeaveOneFileAndNoLeftovers(testCase)
            folder = localScratchFolder(testCase);
            destination = fullfile(folder, 'summary.csv');

            writeTableAtomic(table((1:3).', 'VariableNames', {'value'}), ...
                destination);
            writeTableAtomic(table((4:6).', 'VariableNames', {'value'}), ...
                destination);

            testCase.verifyEqual(readtable(destination).value, (4:6).');
            testCase.verifyEqual(localFileNames(folder), {'summary.csv'});
        end

        function anotherWritersStagingFileIsNeverTouched(testCase)
            % The shared names a second writer would use.
            % Neither helper may delete them, write through them or rename them into place: they hold another process's half-written data.
            folder = localScratchFolder(testCase);
            mat_destination = fullfile(folder, 'checkpoint.mat');
            csv_destination = fullfile(folder, 'summary.csv');
            foreign = { ...
                [mat_destination, '.tmp.mat'], ...
                [mat_destination, '.tmp'], ...
                [csv_destination, '.tmp']};
            for k = 1:numel(foreign)
                localWriteText(foreign{k}, 'another writer''s data');
            end

            saveStructAtomic(mat_destination, 'payload', struct('value', 1));
            writeTableAtomic(table(1, 'VariableNames', {'value'}), ...
                csv_destination);

            for k = 1:numel(foreign)
                testCase.verifyTrue(isfile(foreign{k}), sprintf( ...
                    '%s was removed by a writer that does not own it.', ...
                    foreign{k}));
                testCase.verifyEqual(fileread(foreign{k}), ...
                    'another writer''s data', sprintf( ...
                    '%s was overwritten by a writer that does not own it.', ...
                    foreign{k}));
            end
            testCase.verifyTrue(isfile(mat_destination));
            testCase.verifyTrue(isfile(csv_destination));
        end
    end
end


function folder = localScratchFolder(testCase)
    % A fresh temporary folder that the test framework removes after the test.
    fixture = testCase.applyFixture( ...
        matlab.unittest.fixtures.TemporaryFolderFixture);
    folder = fixture.Folder;
end


function names = localFileNames(folder)
    % The folder's files, sorted, directories excluded.
    listing = dir(folder);
    names = sort({listing(~[listing.isdir]).name});
end


function localWriteText(path, text)
    fid = fopen(path, 'w');
    assert(fid > 0, 'atomicFileWritersTest:CannotWrite', ...
        'could not create the scratch file %s', path);
    closer = onCleanup(@() fclose(fid));
    fprintf(fid, '%s', text);
end
