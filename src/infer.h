#pragma once
#include "logic.h"

#ifdef __cplusplus
extern "C" {
#endif

#define IMG_W 320
#define IMG_H 240
#define IMG_BYTES (IMG_W * IMG_H * 3)

int infer_init(void);                                  /* 0 = ok */
int infer_detect(const uint8_t *rgb, det_t *out, int max);   /* rgb 320x240 RGB888, tra ve so box, <0 loi */
int infer_intent(const char *norm_q);                  /* index INTENT_NAMES, <0 loi */

#ifdef __cplusplus
}
#endif
