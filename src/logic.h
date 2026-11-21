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

#ifdef __cplusplus
}
#endif
