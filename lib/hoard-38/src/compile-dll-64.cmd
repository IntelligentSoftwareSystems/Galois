@echo off
echo Building libhoard.dll and libhoard.lib (ignore linker warnings).
: We need to execute this if we change libhoard.cpp.
@echo off
cl /Iheaplayers /Iheaplayers/util /c /MD /DNDEBUG /Ox /Zp8 libhoard.cpp
cl /Zi /LD libhoard.obj /o libhoard.dll /Ox /link /def:libhoard-64.def /force:multiple kernel32.lib /subsystem:console
: Embed the manifest in the DLL.
mt -manifest libhoard.dll.manifest -outputresource:libhoard.dll;2
