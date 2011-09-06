data = dlmread('mesh.csv', ',', 1, 0);
x = data (:,1);
y = data (:,2);
z = data (:,3);

gridsize = 50;

xlin = linspace (min(x), max(x), gridsize);
ylin = linspace (min(y), max(y), gridsize);

[X, Y] = meshgrid (xlin, ylin);

f = TriScatteredInterp (x,y,z);;

Z = f(X,Y);

hidden off

mesh(X,Y,Z);

figure()

surf(X,Y,Z);

color('interp')
lighting('phong')
