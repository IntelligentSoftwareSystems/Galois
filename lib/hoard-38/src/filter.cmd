REM filter.cmd
@echo off
nm -g libhoard.obj > @@@.@@@ 2>NUL
grep ' T ?' @@@.@@@ > @@@.@@1 2>NUL
grep ' T \_' @@@.@@@ > @@@.@@2 2>NUL
echo EXPORTS > %1
sed 's/.* T //' @@@.@@1 | grep -v DllMain >> %1 2>NUL
sed 's/.* T \_//' @@@.@@2 | grep -v DllMain >> %1 2>NUL
erase @@@.@@@
erase @@@.@@1
erase @@@.@@2
