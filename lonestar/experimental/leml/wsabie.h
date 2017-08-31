#ifndef WSABIE_H
#define WSABIE_H

#include "bilinear.h"
#include "dmat.h"
#include "smat.h"
#include "multilabel.h"

#ifdef __cplusplus
extern "C" {
#endif

double get_wsabieC(const multilabel_parameter *param);

// Project 
void wsabie_model_projection(double *W, double *H, long nr_feats, long nr_labels, long k, double wsabieC);

// nr_samples = 0 => nr_samples = # postive labels in Y
void wsabie_updates(multilabel_problem *prob, multilabel_parameter *param, double *W, double *H, long nr_samples=0);
void wsabie_updates_new(multilabel_problem *prob, multilabel_parameter *param, double *W, double *H, long nr_samples=0);
void wsabie_updates_new2(multilabel_problem *prob, multilabel_parameter *param, double *W, double *H, long nr_samples=0);
void wsabie_updates_new3(multilabel_problem *prob, multilabel_parameter *param, double *W, double *H, long nr_samples=0);
void wsabie_updates_new4(multilabel_problem *prob, multilabel_parameter *param, double *W, double *H, long nr_samples=0);
void wsabie_updates_3(multilabel_problem *prob, multilabel_parameter *param, double *H, long nr_samples=0);
void wsabie_updates_2(multilabel_problem *prob, multilabel_parameter *param, double *W, double *H, long *perm, int t0, long nr_samples=0);
void wsabie_updates_4(multilabel_problem *prob, multilabel_parameter *param, double *H, long *perm, int t0, long nr_samples=0);

#ifdef __cplusplus
}
#endif


#endif
