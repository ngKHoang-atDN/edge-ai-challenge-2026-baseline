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

/* ===== Model v9b: decode CenterNet stride 4 (= decode_v9b trong edgeai-v9b-train.ipynb) ===== */
typedef struct { float s; int cell; int c; } v9b_cand_t;

static int by_cand(const void *a, const void *b)
{
    float d = ((const v9b_cand_t *)b)->s - ((const v9b_cand_t *)a)->s;
    return (d > 0) - (d < 0);
}

int decode_v9b(const int8_t *hm, float hm_scale, int hm_zp,
               const int8_t *reg, float reg_scale, int reg_zp,
               float thr, float iou, det_t *out, int max)
{
    enum { CELLS = V9B_GH * V9B_GW };
    static v9b_cand_t cand[V9B_CAP];
    static det_t tmp[V9B_CAP];
    if (thr <= 0.f || thr >= 1.f) return 0;

    /* nguong logit -> nguong luong tu: giu o co logit >= ln(thr/(1-thr)) */
    float lt = logf(thr / (1.f - thr));
    int qt = (int)ceilf(lt / hm_scale + (float)hm_zp);
    if (qt < -128) qt = -128;
    if (qt > 127) return 0;

    /* histogram int8 de tim nguong q_cut sao cho co <= V9B_CAP o diem cao nhat (khong can sap xep 33600 o) */
    int hist[256] = {0};
    for (int i = 0; i < CELLS * NUM_CLASSES; i++) if (hm[i] >= qt) hist[hm[i] + 128]++;
    int acc = 0, q_cut = qt;
    for (int q = 127; q >= qt; q--) {
        acc += hist[q + 128];
        if (acc >= V9B_CAP) { q_cut = q; break; }
    }
    int n = 0;
    for (int i = 0; i < CELLS * NUM_CLASSES && n < V9B_CAP; i++) {
        if (hm[i] < q_cut) continue;
        float logit = ((int)hm[i] - hm_zp) * hm_scale;
        cand[n++] = (v9b_cand_t){sig(logit), i / NUM_CLASSES, i % NUM_CLASSES};
    }
    qsort(cand, n, sizeof(v9b_cand_t), by_cand);

    int m = 0;
    for (int k = 0; k < n; k++) {
        int cell = cand[k].cell, y = cell / V9B_GW, x = cell % V9B_GW;
        const int8_t *r = reg + cell * 4;
        float r0 = ((int)r[0] - reg_zp) * reg_scale, r1 = ((int)r[1] - reg_zp) * reg_scale;
        float r2 = ((int)r[2] - reg_zp) * reg_scale, r3 = ((int)r[3] - reg_zp) * reg_scale;
        float w = fmaxf(r0, 0.05f) * 10.f * V9B_STRIDE, h = fmaxf(r1, 0.05f) * 10.f * V9B_STRIDE;
        float cx = (x + clamp01(r2)) * V9B_STRIDE, cy = (y + clamp01(r3)) * V9B_STRIDE;
        float x1 = fmaxf(cx - w / 2, 0.f), y1 = fmaxf(cy - h / 2, 0.f);
        float x2 = fminf(cx + w / 2, (float)V9B_W), y2 = fminf(cy + h / 2, (float)V9B_H);
        if (x2 <= x1 || y2 <= y1) continue;
        tmp[m++] = (det_t){cand[k].c, cand[k].s, x1 / V9B_W, y1 / V9B_H, x2 / V9B_W, y2 / V9B_H};
    }
    /* NMS theo lop (tmp da sap giam dan theo diem) */
    int keep = 0;
    for (int i = 0; i < m && keep < max; i++) {
        int ok = 1;
        for (int j = 0; j < keep && ok; j++)
            if (out[j].cls == tmp[i].cls && box_iou(&out[j], &tmp[i]) > iou) ok = 0;
        if (ok) out[keep++] = tmp[i];
    }
    return keep;
}

/* ===== Model v9b: ma hoa cau hoi = 96 ma ky tu + 24 ma bam tu (FNV-1a % 2048 + 1) ===== */
static uint32_t fnv1a(const char *s, size_t n)
{
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 16777619u; }
    return h;
}

void tokenize_v9b(const char *norm, int32_t ids[V9B_SEQ_LEN])
{
    enum { ML = 96, MLW = 24, MAXT = 400 };
    char t[MAXT + 1];               /* chuoi chuan hoa day du (ma tu lay tu day du; ma ky tu chi lay 96 ky tu dau) */
    int len = 0, pending_space = 0;
    memset(ids, 0, V9B_SEQ_LEN * sizeof(int32_t));
    for (const char *p = norm; *p && len < MAXT - 1; p++) {
        char c = *p;
        if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
            if (pending_space && len > 0) t[len++] = ' ';
            pending_space = 0;
            t[len++] = c;
        } else {
            pending_space = 1;      /* moi ky tu khac (ke ca '_') = khoang trang; bo khoang trang dau/cuoi/lap */
        }
    }
    t[len] = 0;
    /* ma ky tu: ' '->2, 'a'..'z'->3..28, '0'..'9'->29..38 (CH = " abcdefghijklmnopqrstuvwxyz0123456789") */
    for (int i = 0; i < len && i < ML; i++) {
        char c = t[i];
        ids[i] = c == ' ' ? 2 : (c <= '9' ? 29 + (c - '0') : 3 + (c - 'a'));
    }
    /* ma tu: toi da 24 tu */
    int w = 0;
    for (int i = 0; i < len && w < MLW;) {
        int j = i;
        while (j < len && t[j] != ' ') j++;
        if (j > i) ids[ML + w++] = (int32_t)(fnv1a(t + i, (size_t)(j - i)) % 2048u + 1u);
        i = j + 1;
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
