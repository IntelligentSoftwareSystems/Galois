#pragma once

struct ggc_rt_dev_info {
  int dev;
  int nSM;
};

void ggc_init_dev_info();
void ggc_set_gpu_device(int dev);
int ggc_get_nSM();
