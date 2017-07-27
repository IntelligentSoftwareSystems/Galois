CLANG_LEVEL := ../..
LIBRARYNAME = OpenCLCodeGenAction

# If we don't need RTTI or EH, there's no reason to export anything
# from the plugin.
ifneq ($(REQUIRES_RTTI), 1)
ifneq ($(REQUIRES_EH), 1)
EXPORTED_SYMBOL_FILE = $(PROJ_SRC_DIR)/OpenCLCodeGen.exports
endif
endif

LINK_LIBS_IN_SHARED = 0
LOADABLE_MODULE = 1

include $(CLANG_LEVEL)/Makefile

ifeq ($(OS),Darwin)
  LDFLAGS=-Wl,-undefined,dynamic_lookup
endif
