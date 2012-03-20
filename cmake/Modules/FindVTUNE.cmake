set(VTUNE_INCLUDE_DIRS "/opt/intel/vtune_amplifier_xe_2011/include")
if(EXISTS ${VTUNE_INCLUDE_DIRS})
  set(VTUNE_FOUND_INTERNAL "YES")
  set(VTUNE_LIBRARIES "/opt/intel/vtune_amplifier_xe_2011/lib64/libittnotify.a" dl)
else()
  set(VTUNE_FOUND off)
  set(VTUNE_INCLUDE_DIRS off)
endif()

find_package_handle_standard_args(VTUNE DEFAULT_MSG VTUNE_FOUND_INTERNAL)

