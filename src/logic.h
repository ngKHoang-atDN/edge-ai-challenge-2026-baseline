/* Logic thuan C (khong phu thuoc IDF) -> test duoc tren may host: test/test_logic.c */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NUM_CLASSES 7   /* thu tu = classes.txt cua dataset */
#define NUM_INTENTS 5   /* sorted(df.intent.unique()) trong baseline.ipynb */
#define SEQ_LEN     32

extern const char *const CLASS_NAMES[NUM_CLASSES];
extern const char *const INTENT_NAMES[NUM_INTENTS];

typedef struct { int cls; float score, x1, y1, x2, y2; } det_t;   /* toa do chuan hoa 0..1 */

/* g: output da dequantize, layout NHWC [H][W][5+NUM_CLASSES] = obj, tx, ty, w, h, cls logits */
int decode_grid(const float *g, int H, int W, float conf, float iou, det_t *out, int max);

/* lowercase, ky tu khong phai [a-z0-9_] -> ' ' (giong re.findall(r"\w+") tren chuoi khong dau) */
void normalize_text(char *s);

/* vocab[k] co id k+2; 0 = pad, 1 = unknown */
void tokenize(const char *norm, const char *const *vocab, int nv, int32_t ids[SEQ_LEN]);

void make_answer(int intent, const char *norm_q, const det_t *d, int n, char *out, size_t sz);

/* ===== Model v9b cua doi (CenterNet stride 4 + TextCNN hash) - xem edgeai-v9b-train.ipynb ===== */
#define V9B_W      320
#define V9B_H      240
#define V9B_GW     80      /* luoi 60x80 */
#define V9B_GH     60
#define V9B_STRIDE 4
#define V9B_CAP    600     /* so o diem cao nhat xet truoc NMS */
#define V9B_SEQ_LEN 120    /* 96 ma ky tu + 24 ma bam tu */

/* hm: int8 [60][80][NUM_CLASSES] (logit heatmap), reg: int8 [60][80][4] (w, h, off_x, off_y).
 * scale/zp lay tu tensor->params. thr = nguong diem sau sigmoid, iou = nguong NMS theo lop.
 * Tra ve so box (<= max), da sap giam dan theo diem; toa do chuan hoa 0..1 (nhu det_t). */
int decode_v9b(const int8_t *hm, float hm_scale, int hm_zp,
               const int8_t *reg, float reg_scale, int reg_zp,
               float thr, float iou, det_t *out, int max);

/* norm: cau hoi da bo dau (chu thuong/hoa, ky tu la -> khoang trang deu duoc). */
void tokenize_v9b(const char *norm, int32_t ids[V9B_SEQ_LEN]);

#ifdef __cplusplus
}
#endif
