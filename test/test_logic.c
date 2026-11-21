/* xem README */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "logic.h"

#define H 15
#define W 20
#define C (5 + NUM_CLASSES)

static void put(float *g, int y, int x, int cls, float obj)
{
    float *v = g + (y * W + x) * C;
    v[0] = obj; v[1] = v[2] = 0; v[3] = v[4] = 0;    /* offset 0.5, w=h=0.5 */
    v[5 + cls] = 10;
}

int main(void)
{
    static float g[H * W * C];
    for (int i = 0; i < H * W; i++) g[i * C] = -10;   /* obj ~ 0 */
    put(g, 7, 2, 1, 5);          /* cup ben trai */
    put(g, 7, 3, 1, 4);          /* cup chong len cai tren -> NMS bo */
    put(g, 7, 17, 1, 5);         /* cup ben phai */
    put(g, 2, 10, 3, 5);         /* book giua */
    det_t d[64];
    int n = decode_grid(g, H, W, 0.3f, 0.5f, d, 64);
    assert(n == 3);
    assert(d[0].score >= d[1].score && d[1].score >= d[2].score);

    const char *vocab[] = {"bao", "co", "coc", "nhieu"};
    char q[] = "Co bao nhieu COC xyz?";   /* web da bo dau truoc khi gui */
    normalize_text(q);
    int32_t ids[SEQ_LEN];
    tokenize(q, vocab, 4, ids);
    assert(ids[0] == 3 && ids[1] == 2 && ids[2] == 5 && ids[3] == 4 && ids[4] == 1 && ids[5] == 0);

    char a[256];
    char q1[] = "co bao nhieu coc"; make_answer(0, q1, d, n, a, sizeof a); puts(a); assert(strstr(a, "2"));
    char q2[] = "co dien thoai khong"; make_answer(1, q2, d, n, a, sizeof a); puts(a); assert(strstr(a, "Không"));
    char q3[] = "sach nam o dau"; make_answer(3, q3, d, n, a, sizeof a); puts(a); assert(strstr(a, "giữa"));
    make_answer(2, q1, d, n, a, sizeof a); puts(a); assert(!strcmp(a, "Có 2 loại đối tượng: cốc, sách."));
    make_answer(4, q1, d, n, a, sizeof a); puts(a);
    char q4[] = "hop co khong phu"; make_answer(1, q4, d, 0, a, sizeof a); puts(a);
    puts("OK");
    return 0;
}
