%ENABLEFIGUREAUTOMAXIMIZE Maximize every figure created for the rest of the MATLAB session.
%
%   listener_handle = enableFigureAutoMaximize()
%
% Sets the groot figure defaults WindowStyle = 'normal' and WindowState = 'maximized' where the release accepts them.
% Installs a groot 'ObjectChildAdded' listener that puts each newly created figure in a standalone window and then sets WindowState = 'maximized'.
% R2025a and R2025b create figures as tabs in a figure container by default, and in that docked state setting WindowState can undock the figure but leave it normal-sized; setting WindowStyle first avoids this.
%
% Returns the listener handle, so a caller can stop the behaviour with delete(listener_handle).
% Normal use discards it: the listener then lives until clear all, clear classes or a MATLAB restart.
%
% Safety:
%   * Only matlab.ui.Figure objects are maximized.
%   * Any failure inside the listener (an App Designer uifigure, a session without a display, a figure already closed) is caught silently, so the listener never breaks the analysis.

function listener_handle = enableFigureAutoMaximize()

    try
        set(groot, 'DefaultFigureWindowStyle', 'normal');
        set(groot, 'DefaultFigureWindowState', 'maximized');
    catch
        % Some MATLAB releases and sessions without a display reject these window defaults; the listener below still handles ordinary desktop figures.
    end

    listener_handle = addlistener(groot, 'ObjectChildAdded', ...
        @(~, evt) localMaximizeIfFigure(evt.Child));

end


function localMaximizeIfFigure(h)
    % Undock and maximize h when it is a figure.
    if ~isa(h, 'matlab.ui.Figure')
        return;
    end
    try
        h.WindowStyle = 'normal';
        drawnow limitrate;
        h.WindowState = 'maximized';
    catch
        % Ignored: this happens without a display, for some uifigure types, and for a figure closed before this assignment.
    end
end
