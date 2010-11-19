@echo off
echo Building winhoard.dll.
cl /W0 /MD /Iheaplayers /Iheaplayers/util /nologo /Ob2 /Ox /Zp8 /DNDEBUG /D_MT /c winhoard.cpp
cl /W0 /MD /Iheaplayers /Iheaplayers/util /nologo /Ob2 /Ox /Zp8 /DNDEBUG /D_MT /c usewinhoard.cpp
cl /Zi /MD /LD winhoard.obj /o winhoard.dll /link /force:multiple /base:0x63000000 kernel32.lib msvcrt.lib /subsystem:console /incremental:no /dll /entry:HoardDllMain
: Embed the manifest in the DLL.
mt -manifest winhoard.dll.manifest -outputresource:winhoard.dll;2
echo *****
echo Build complete.

