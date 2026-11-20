classdef settingsIdentityTest < matlab.unittest.TestCase
    % SETTINGSIDENTITYTEST The settings identity every resume is checked against.
    %
    % Every phase stamps this digest into what it writes and refuses to reuse or merge an artifact whose stamp is not the one the current settings produce.
    % Two properties therefore have to hold exactly: the identity must not move when nothing scientific changes (field order, the class a flag happens to carry), and it must move when anything scientific does (a grid, a size, a fit option, the upstream phase's identity).
    %
    % The suite uses synthetic structs only and reads no recordings or results.

    methods (TestClassSetup)

        function addSourceTree(testCase)
            sourceTree = fullfile( ...
                                  fileparts(fileparts(mfilename('fullpath'))), 'src');
            assert(isfolder(sourceTree), ...
                   'settingsIdentityTest:SourceTreeNotFound', ...
                   'expected this file under <repo>/tests/; got %s', ...
                   mfilename('fullpath'));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                                                                       sourceTree, IncludingSubfolders = true));
        end

    end

    methods (Test)

        function digestIsASha256Hex(testCase)
            id = settingsIdentity(localSettings());

            testCase.verifyClass(id, 'char');
            testCase.verifyMatches(id, '^[0-9a-f]{64}$', ...
                                   'the identity is a lowercase 64-character SHA-256 digest.');
        end

        function fieldOrderDoesNotMove(testCase)
            settings = localSettings();
            reordered = orderfields(settings, flip(fieldnames(settings)));
            reordered.fit_opts = orderfields(reordered.fit_opts, ...
                                             flip(fieldnames(reordered.fit_opts)));

            testCase.verifyEqual(settingsIdentity(reordered), ...
                                 settingsIdentity(settings), ...
                                 ['the same settings written in another field order are the ' ...
                                  'same science and must share one identity.']);
        end

        function flagClassDoesNotMove(testCase)
            settings = localSettings();
            settings.fit_opts.store_profile = true;
            as_double = settings;
            as_double.fit_opts.store_profile = 1;

            testCase.verifyEqual(settingsIdentity(as_double), ...
                                 settingsIdentity(settings), ...
                                 'a logical and the 0/1 it stands for are the same setting.');
        end

        function everyScientificFieldMovesIt(testCase)
            settings = localSettings();
            reference = settingsIdentity(settings);

            changes = {@(s) setfield(s, 'd_list', 3:19), ...
                       @(s) setfield(s, 'groups_per_d', 11), ...
                       @(s) setfield(s, 'max_pairs', 51), ...
                       @(s) setfield(s, 'profile_grid', linspace(0, 3, 21)), ...
                       @(s) setfield(s, 'upstream_identity', repmat('b', 1, 64)), ...
                       @(s) localSetFitOption(s, 'tol_x', 1e-7), ...
                       @(s) localSetFitOption(s, 'p_bounds', [0, 2])};
            for k = 1:numel(changes)
                changed = changes{k}(settings);
                testCase.verifyNotEqual(settingsIdentity(changed), reference, ...
                                        sprintf('change %d must produce a different identity.', k));
            end
        end

        function emptyAndMaterialisedDefaultsDiffer(testCase)
            % The reason the caller must materialise: an empty override struct is not the options it stands for, and hashing it would let two different fit configurations share one identity.
            placeholder = localSettings();
            placeholder.fit_opts = struct();

            testCase.verifyNotEqual(settingsIdentity(placeholder), ...
                                    settingsIdentity(localSettings()), ...
                                    'an unmaterialised override struct is not its defaults.');
        end

        function structArraysAndCellsAreSupported(testCase)
            settings = localSettings();
            settings.designs = [struct('name', 'equi', 'rho', 0.4), ...
                                struct('name', 'block', 'rho', 0.75)];
            settings.labels = {'recovery', 'factorial'};

            first = settingsIdentity(settings);
            settings.designs(2).rho = 0.76;

            testCase.verifyMatches(first, '^[0-9a-f]{64}$');
            testCase.verifyNotEqual(settingsIdentity(settings), first, ...
                                    'a value inside a struct array must reach the identity.');
        end

    end
end

function settings = localSettings()
    % One phase's identity fields, shaped like Phase 2's.
    settings = struct( ...
                      'phase',             'Phase2', ...
                      'd_list',            3:20, ...
                      'groups_per_d',      10, ...
                      'max_pairs',         50, ...
                      'fit_opts',          tFitDefaultOptions(), ...
                      'profile_grid',      linspace(0, 3, 41), ...
                      'upstream_identity', repmat('a', 1, 64));
end

function settings = localSetFitOption(settings, name, value)
    settings.fit_opts.(name) = value;
end
