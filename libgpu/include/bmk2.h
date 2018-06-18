#pragma once

#ifdef __cplusplus
extern "C" {
#endif
char* bmk2_get_binid();
char* bmk2_get_inputid();
char* bmk2_get_runid();
int bmk2_log_collect(const char* component, const char* file);

#ifdef __cplusplus
}
#endif
