poly = dlmread ('mesh-poly.csv', ',' , 1, 0);

coord = dlmread ('mesh-coord.csv', ',', 1, 0);

xcoord = coord(:, 1);

ycoord = coord(:, 2);

zcoord = coord(:, 3);

t = poly (:, 1:4);

t = t + 1;

% triplot (t, x, y);

timestamps = poly(:, 4);

norm_ts = timestamps ./ max(timestamps); % scale relative to 1;

norm_freq = min (timestamps) ./ timestamps;


sz = size(t)

% colormap Gray;
colormap ('default');

cbar = colorbar ('location', 'EastOutside');
set( get(cbar, 'Ylabel'), 'String', 'Number of updates per element normalized to 1');


hold on;
for i = 1 : sz(1)
    x = xcoord(t(i,:));
    y = ycoord(t(i,:));
    z = zcoord(t(i,:));
    
    fill3 (x, y, z, norm_freq(i), 'LineWidth', 2);
    
end

set (get(gca, 'Xlabel'), 'String', 'Xpos of elements');
set (get(gca, 'Ylabel'), 'String', 'Ypos of elements');
title ('Mesh elements colored by number of updates');

