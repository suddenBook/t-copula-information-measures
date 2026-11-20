% MERGEOPTIONS Fill missing option fields from defaults.
%
% A field already in options keeps its value and a field only in defaults is added; nested structs are not merged.
%
% Inputs:
%   defaults - struct of default values.
%   options  - caller-supplied struct.
%
% Outputs:
%   options  - options with every missing field of defaults added.

function options = mergeOptions(defaults, options)

    names = fieldnames(defaults);
    for i = 1:length(names)
        if ~isfield(options, names{i})
            options.(names{i}) = defaults.(names{i});
        end
    end

end
