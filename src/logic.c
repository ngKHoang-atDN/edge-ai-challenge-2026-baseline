#include "logic.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *const CLASS_NAMES[NUM_CLASSES] = {"bottle", "cup", "phone", "book", "mouse", "box", "toy_car"};
const char *const INTENT_NAMES[NUM_INTENTS] = {"COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"};

static const char *const VN[NUM_CLASSES] = {"chai", "cốc", "điện thoại", "sách", "chuột", "hộp", "xe đồ chơi"};
/* tu khoa (khong dau) de biet cau hoi nhac toi class nao */
static const char *const KW[NUM_CLASSES][4] = {
    {"chai"}, {"coc", "tach", "ly"}, {"thoai", "smartphone"}, {"sach"},
    {"chuot"}, {"hop", "thung"}, {"xe", "oto"},
};

static float sig(float x) { return 1.f / (1.f + expf(-x)); }
static float clamp01(float x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

static int by_score(const void *a, const void *b)
{
    float d = ((const det_t *)b)->score - ((const det_t *)a)->score;
    return (d > 0) - (d < 0);
}

static float box_iou(const det_t *a, const det_t *b)
{
    float w = fminf(a->x2, b->x2) - fmaxf(a->x1, b->x1);
    float h = fminf(a->y2, b->y2) - fmaxf(a->y1, b->y1);
    if (w <= 0 || h <= 0) return 0;
    float inter = w * h;
    float u = (a->x2 - a->x1) * (a->y2 - a->y1) + (b->x2 - b->x1) * (b->y2 - b->y1) - inter;
    return u > 0 ? inter / u : 0;
}

/* = decode_boxes + softmax + batched_nms trong baseline.ipynb */
int decode_grid(const float *g, int H, int W, float conf, float iou, det_t *out, int max)
{
    const int C = 5 + NUM_CLASSES;
    int n = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            const float *v = g + (y * W + x) * C;
            int best = 0;
            for (int k = 1; k < NUM_CLASSES; k++) if (v[5 + k] > v[5 + best]) best = k;
            float sum = 0;
            for (int k = 0; k < NUM_CLASSES; k++) sum += expf(v[5 + k] - v[5 + best]);
            float score = sig(v[0]) / sum;
            if (score < conf || n >= max) continue;
            float cx = (sig(v[1]) + x) / W, cy = (sig(v[2]) + y) / H, w = sig(v[3]), h = sig(v[4]);
            out[n++] = (det_t){best, score, clamp01(cx - w / 2), clamp01(cy - h / 2),
                               clamp01(cx + w / 2), clamp01(cy + h / 2)};
        }
    qsort(out, n, sizeof(det_t), by_score);
    int keep = 0;
    for (int i = 0; i < n; i++) {
        int ok = 1;
        for (int j = 0; j < keep && ok; j++)
            if (out[j].cls == out[i].cls && box_iou(&out[j], &out[i]) > iou) ok = 0;
        if (ok) out[keep++] = out[i];
    }
    return keep;
}

void normalize_text(char *s)
{
    for (; *s; s++) {
        char c = *s;
        if (c >= 'A' && c <= 'Z') *s = c - 'A' + 'a';
        else if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) *s = ' ';
    }
}

void tokenize(const char *norm, const char *const *vocab, int nv, int32_t ids[SEQ_LEN])
{
    int n = 0;
    memset(ids, 0, SEQ_LEN * sizeof(int32_t));
    while (*norm && n < SEQ_LEN) {
        while (*norm == ' ') norm++;
        size_t len = strcspn(norm, " ");
        if (!len) break;
        int id = 1;
        for (int k = 0; k < nv; k++)
            if (strlen(vocab[k]) == len && !memcmp(vocab[k], norm, len)) { id = k + 2; break; }
        ids[n++] = id;
        norm += len;
    }
}

static int has_word(const char *norm, const char *w)
{
    size_t len = strlen(w);
    for (const char *p = norm; (p = strstr(p, w)); p++)
        if ((p == norm || p[-1] == ' ') && (p[len] == ' ' || p[len] == 0)) return 1;
    return 0;
}

static int find_class(const char *norm)
{
    for (int c = 0; c < NUM_CLASSES; c++)
        for (int k = 0; k < 4 && KW[c][k]; k++)
            if (has_word(norm, KW[c][k])) return c;
    return -1;
}

void make_answer(int intent, const char *norm_q, const det_t *d, int n, char *out, size_t sz)
{
    const char *name = INTENT_NAMES[intent];
    if (!strcmp(name, "OUT_OF_SCOPE")) {
        snprintf(out, sz, "Câu hỏi nằm ngoài phạm vi.");
        return;
    }
    if (!strcmp(name, "LIST")) {
        int seen[NUM_CLASSES] = {0}, k = 0;
        for (int i = 0; i < n; i++) k += !seen[d[i].cls]++;
        size_t p = snprintf(out, sz, "Có %d loại đối tượng", k);
        const char *sep = ":";
        for (int c = 0; c < NUM_CLASSES && p < sz; c++)
            if (seen[c]) { p += snprintf(out + p, sz - p, "%s %s", sep, VN[c]); sep = ","; }
        if (p < sz) snprintf(out + p, sz - p, ".");
        return;
    }
    int c = find_class(norm_q);
    if (c < 0) {
        snprintf(out, sz, "Không xác định được đối tượng trong câu hỏi.");
        return;
    }
    int cnt = 0, best = -1;
    for (int i = 0; i < n; i++)
        if (d[i].cls == c) { cnt++; if (best < 0) best = i; }   /* d da sort theo score */
    if (!strcmp(name, "COUNT")) {
        snprintf(out, sz, "Có %d %s.", cnt, VN[c]);
    } else if (!strcmp(name, "EXIST")) {
        snprintf(out, sz, cnt ? "Có %s trong ảnh." : "Không có %s trong ảnh.", VN[c]);
    } else if (best < 0) {
        snprintf(out, sz, "Không thấy %s trong ảnh.", VN[c]);
    } else {
        float cx = (d[best].x1 + d[best].x2) / 2;
        snprintf(out, sz, "%s nằm %s.", VN[c], cx < 1.f / 3 ? "bên trái" : cx < 2.f / 3 ? "ở giữa" : "bên phải");
    }
}
