%SAVEFIGUREASPDF Save the current figure as a vector PDF and an editable .fig with the same base name.
%
%   saveFigureAsPDF(filename, width, height)
%
% Uses exportgraphics (R2020a or later) for vector output on a transparent background.
% With width, and optionally height, in inches the figure is resized first; a missing height keeps the aspect ratio.
% Without them the figure is exported as it is.

function saveFigureAsPDF(filename, width, height)

    fig = gcf;

    if nargin >= 2
        fig.Units = 'inches';
        if nargin < 3
            height = width * (fig.Position(4) / fig.Position(3));  % preserve aspect ratio
        end
        fig.Position(3:4) = [width, height];
    end

    exportgraphics(fig, filename, 'ContentType', 'vector', 'BackgroundColor', 'none');

    [folder, name, ~] = fileparts(filename);
    savefig(fig, fullfile(folder, [name, '.fig']));

end
