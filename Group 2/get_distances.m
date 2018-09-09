function [dh, dv] = get_distances(x, y)
%GET_DISTANCES Return the vertical and horizontal distance from obstacles
%   This function calculates the distance of the car with regards to the
%   distance from the obstacle in the area and returns dh and dv

% We split cases depending on where the car is on the map
% 4 different cases, depending on the x position

if (x <= 5)
    dv = y;
    if (y <= 1)
        dh = 5 - x;
    elseif (y <= 2)
        dh = 6 - x;
    elseif (y <= 3)
        dh = 7 - x;
    else
        dh = 11 - x;
    end
elseif (x <= 6)
    dv = y - 1;
    if (y <= 2)
        dh  = 6 - x;
    elseif (y <= 3)
        dh = 7 - x;
    else
        dh = 11 - x;
    end
elseif (x <= 7)
    dv = y - 2;
    if (y <= 3)
        dh = 7 - x;
    else
        dh = 11 - x;
    end
elseif (x <= 10)
    dv = y - 3;
    dh = 11 - x;
end

% maximum value of distance is one
if (dv>1)
    dv = 1;
end
if (dh>1)
    dh=1;
end

end

