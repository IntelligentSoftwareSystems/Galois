# Commands to compile Hoard for various targets.
# Run make (with no arguments) to see the complete target list.

FREEBSD_COMPILE := g++ -O2 -DNDEBUG  -DNDEBUG  -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -shared libhoard.cpp -Bsymbolic -o libhoard.so -pthread -fPIC

# This is the "macwrapper" version, but the wholesale replacement of malloc works better...
# MACOS_COMPILE := g++ -pipe -O2 -DNDEBUG  -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -compatibility_version 1 -current_version 1 -dynamiclib -D'CUSTOM_PREFIX(x)=hoard\#\#x' macwrapper.cpp libhoard.cpp -o libhoard.dylib -ldl -lpthread

MACOS_COMPILE := g++ -Wall -pedantic -Wextra -Wchar-subscripts -Wcomment -Wformat -Winit-self -Wuninitialized -pipe -O2 -DNDEBUG  -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -compatibility_version 1 -current_version 1 -dynamiclib libhoard.cpp -o libhoard.dylib -ldl -lpthread

MACOS_COMPILE_DEBUG := g++ -pipe -g -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -compatibility_version 1 -current_version 1 -dynamiclib libhoard.cpp -o libhoard.dylib -ldl -lpthread

# MACOS_COMPILE_DEBUG := g++ -pipe -g -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -compatibility_version 1 -current_version 1 -dynamiclib -D'CUSTOM_PREFIX(x)=hoard\#\#x' macwrapper.cpp libhoard.cpp -o libhoard.dylib -ldl -lpthread

LINUX_GCC_x86_COMPILE := g++ -m32 -I/usr/include/nptl -ffast-math -g -malign-double -pipe -march=pentium4 -O3 -finline-limit=20000 -fomit-frame-pointer -finline-functions  -DNDEBUG  -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -shared -D'CUSTOM_PREFIX(x)=hoard\#\#x' gnuwrapper.cpp libhoard.cpp -Bsymbolic -o libhoard.so -ldl -lpthread

# LINUX_GCC_x86_COMPILE := g++ -m32 -I/usr/include/nptl -ffast-math -g -malign-double -pipe -march=pentium4 -O3 -finline-limit=20000 -fomit-frame-pointer -finline-functions  -DNDEBUG  -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -shared libhoard.cpp -Bsymbolic -o libhoard.so -ldl -lpthread

LINUX_GCC_x86_64_COMPILE := g++ -W -Wconversion -Wall -march=nocona -m64 -msse2 -I/usr/include/nptl -pipe -fPIC -O3 -finline-limit=20000 -fomit-frame-pointer -finline-functions  -DNDEBUG  -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -shared  -D'CUSTOM_PREFIX(x)=hoard\#\#x' gnuwrapper.cpp libhoard.cpp -Bsymbolic -o libhoard.so -ldl -lpthread

LINUX_GCC_x86_64_COMPILE_DEBUG := g++ -W -Wconversion -Wall -march=nocona -m64 -I/usr/include/nptl -pipe -fPIC -O0 -g -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -shared  -D'CUSTOM_PREFIX(x)=hoard\#\#x' gnuwrapper.cpp libhoard.cpp -Bsymbolic -o libhoard.so -ldl -lpthread

LINUX_GCC_x86_COMPILE_STATIC := g++ -m32 -I/usr/include/nptl -static -malign-double -pipe -march=pentium4 -O3 -finline-limit=20000 -fomit-frame-pointer -finline-functions  -DNDEBUG  -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -D'CUSTOM_PREFIX(x)=hoard\#\#x' -c gnuwrapper.cpp libhoard.cpp ; ar cr libhoard.a libhoard.o

LINUX_GCC_x86_COMPILE_DEBUG := g++ -m32 -O0 -fno-inline -malign-double -I/usr/include/nptl  -g -pipe -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -D'CUSTOM_PREFIX(x)=hoard\#\#x' -shared gnuwrapper.cpp libhoard.cpp -Bsymbolic -o libhoard.so -ldl -lpthread

SOLARIS_SUNW_SPARC_COMPILE_32_DEBUG := CC -dalign -mt -g -xildoff -xthreadvar=dynamic -L/usr/lib/lwp -R/usr/lib/lwp -DNDEBUG -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -G -PIC libhoard.cpp heaplayers/util/sparc-interchange.il -o libhoard_32.so -lthread -ldl -lCrun

SOLARIS_SUNW_SPARC_COMPILE_32 := CC -dalign -xbuiltin=%all -mt -xO5 -fast -xtarget=ultra2 -xildoff -xthreadvar=dynamic -L/usr/lib/lwp -R/usr/lib/lwp -DNDEBUG -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -G -PIC libhoard.cpp heaplayers/util/sparc-interchange.il -o libhoard_32.so -lthread -ldl -lCrun

SOLARIS_SUNW_SPARC_COMPILE_64 := CC -xcode=pic13 -m64 -mt -fast -dalign -xbuiltin=%all -xO5 -xtarget=ultra2 -xildoff -xthreadvar=dynamic -L/usr/lib/lwp -R/usr/lib/lwp -DNDEBUG -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -G -PIC libhoard.cpp heaplayers/util/sparc-interchange.il -o libhoard_64.so -lthread -ldl -lCrun

SOLARIS_SUNW_x86_COMPILE_32 := CC -fns -fsimple=2 -ftrap=%none -xbuiltin=%all -mt -xO5 -fast -xildoff -xthreadvar=dynamic -L/usr/lib/lwp -R/usr/lib/lwp -DNDEBUG -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -G -PIC libhoard.cpp heaplayers/util/x86-interchange.il -o libhoard_32.so -lthread -ldl -lCrun

SOLARIS_SUNW_x86_COMPILE_32_DEBUG := CC -mt -g -xildoff -xthreadvar=dynamic -L/usr/lib/lwp -R/usr/lib/lwp -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -G -PIC libhoard.cpp heaplayers/util/x86-interchange.il -o libhoard_32.so -lthread -ldl -lCrun

SOLARIS_SUNW_x86_COMPILE_64 := CC -xarch=amd64 -fns -fsimple=2 -ftrap=%none -xbuiltin=%all -xO5 -xildoff -xthreadvar=dynamic -L/usr/lib/lwp -R/usr/lib/lwp -DNDEBUG -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -G -PIC libhoard.cpp heaplayers/util/x86_64-interchange.il -o libhoard_64.so -lthread -ldl -lCrun

# SOLARIS_GCC_SPARC_COMPILE := g++ -nostartfiles -pipe -DNDEBUG -mcpu=ultrasparc -O3 -finline-limit=20000 -fPIC -fkeep-inline-functions -finline-functions -ffast-math -fomit-frame-pointer -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -shared libhoard.cpp -lthread -lpthread -ldl -o libhoard.so

SOLARIS_GCC_SPARC_COMPILE_32 := g++ -nostartfiles -pipe -DNDEBUG -mcpu=ultrasparc -m32 -O3 -finline-limit=20000 -fPIC -fkeep-inline-functions -finline-functions -ffast-math -fomit-frame-pointer -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -shared libhoard.cpp -lthread -lpthread -ldl -o libhoard_32.so

SOLARIS_GCC_SPARC_COMPILE_64 := g++ -nostartfiles -pipe -DNDEBUG -mcpu=ultrasparc -m64 -O3 -finline-limit=20000 -fPIC -fkeep-inline-functions -finline-functions -ffast-math -fomit-frame-pointer -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -shared libhoard.cpp -lthread -lpthread -ldl -o libhoard_64.so

SOLARIS_GCC_SPARC_COMPILE_DEBUG := g++ -nostartfiles -pipe -mcpu=ultrasparc -g -fPIC -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1 -shared libhoard.cpp -lthread -lpthread -ldl -o libhoard.so

GENERIC_GCC_COMPILE := g++ -I/usr/include/nptl  -malign-double -pipe -O3 -finline-limit=20000 -fomit-frame-pointer -finline-functions  -DNDEBUG  -I. -Iheaplayers -Iheaplayers/util -D_REENTRANT=1  -D'CUSTOM_PREFIX(x)=hoard\#\#x' -shared gnuwrapper.cpp libhoard.cpp -Bsymbolic -o libhoard.so -ldl -lpthread

DEPS = libhoard.cpp check.h fixedrequestheap.h hoardmanager.h addheaderheap.h semilockedheap.h alignedheap.h threadpoolheap.h redirectfree.h ignoreinvalidfree.h conformantheap.h hoardsuperblock.h lockmallocheap.h alignedsuperblockheap.h alignedmmap.h globalheap.h

all:
	@echo "To build Hoard, specify the desired build target:"

	@echo " freebsd"
	@echo " linux-gcc-x86"
	@echo " linux-gcc-x86-64"
	@echo " macos"
	@echo " solaris-sunw-sparc"
	@echo " solaris-sunw-x86"
	@echo " solaris-sunw-x86-64"
	@echo " solaris-gcc-sparc"
	@echo " generic-gcc"
	@echo "  (for build instructions on Windows, see NOTES.Windows)"

.PHONY: macos freebsd linux-gcc-x86 linux-gcc-x86-debug solaris-sunw-sparc solaris-sunw-x86 solaris-gcc-sparc generic-gcc linux-gcc-x86-64 clean

macos:
	$(MACOS_COMPILE)

macos-debug:
	$(MACOS_COMPILE_DEBUG)

freebsd:
	$(FREEBSD_COMPILE)

generic-gcc:
	$(GENERIC_GCC_COMPILE)

linux-gcc-x86:
	$(LINUX_GCC_x86_COMPILE)

linux-gcc-x86-static:
	$(LINUX_GCC_x86_COMPILE_STATIC)

linux-gcc-x86-debug:
	$(LINUX_GCC_x86_COMPILE_DEBUG)

linux-gcc-x86-64:
	$(LINUX_GCC_x86_64_COMPILE)

linux-gcc-x86-64-debug:
	$(LINUX_GCC_x86_64_COMPILE_DEBUG)

solaris-sunw-sparc-debug:
	$(SOLARIS_SUNW_SPARC_COMPILE_32_DEBUG)

solaris-sunw-sparc:
	$(SOLARIS_SUNW_SPARC_COMPILE_32)
	$(SOLARIS_SUNW_SPARC_COMPILE_64)

solaris-gcc-sparc:
	$(SOLARIS_GCC_SPARC_COMPILE_32)
	$(SOLARIS_GCC_SPARC_COMPILE_64)

solaris-gcc-sparc-debug:
	$(SOLARIS_GCC_SPARC_COMPILE_DEBUG)

solaris-sunw-x86:
	$(SOLARIS_SUNW_x86_COMPILE_32)

solaris-sunw-x86-64:
	$(SOLARIS_SUNW_x86_COMPILE_64)

clean:
	rm -f libhoard.so


