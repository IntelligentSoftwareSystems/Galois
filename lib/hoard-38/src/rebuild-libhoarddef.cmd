: We need to execute this if we change libhoard.cpp.
@echo off
cl /Iheaplayers -Iheaplayers/util /c /MD /DNDEBUG /Ox /Zp8 /Oy libhoard.cpp
nm -g libhoard.obj > @@@.@@@ 2>NUL
grep ' T ?' @@@.@@@ > @@@.@@1 2>NUL
grep ' T \_' @@@.@@@ > @@@.@@2 2>NUL
echo EXPORTS > libhoard.def
sed 's/.* T //' @@@.@@1 | grep -v DllMain >> libhoard.def 2>NUL
sed 's/.* T \_//' @@@.@@2 | grep -v DllMain >> libhoard.def 2>NUL
erase @@@.@@@
erase @@@.@@1
erase @@@.@@2
 
