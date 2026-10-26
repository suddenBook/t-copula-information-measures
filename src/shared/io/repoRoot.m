function root = repoRoot()
    %REPOROOT Absolute path of the repository root.
    %
    % The root is the nearest folder, walking up from this file's own folder, that contains both src/ and scripts/, so the result does not depend on how deep under the repository this file or its callers sit.
    %
    % It is callable only once src/ is on the path (this file lives under src/shared/io/).
    % Entry scripts, which must put src/ on the path before any src/ function is callable, carry the same walk-up inline instead.
    %
    % Output:
    %   root - char, the repository root folder; an error names this file's folder when no folder above it qualifies.

    root = fileparts(mfilename('fullpath'));
    while ~(isfolder(fullfile(root, 'src')) && isfolder(fullfile(root, 'scripts')))
        parent = fileparts(root);
        if strcmp(parent, root)
            error('repoRoot:NotFound', ...
                'repository root (a directory containing src/ and scripts/) not found above %s', ...
                fileparts(mfilename('fullpath')));
        end
        root = parent;
    end
end
