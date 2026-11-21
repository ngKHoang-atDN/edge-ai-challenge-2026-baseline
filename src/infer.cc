#include "infer.h"
#include <math.h>
#include <string.h>
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"
#include "model_config.h"

static const char *TAG = "infer";

static tflite::MicroInterpreter *s_det, *s_txt;
static int8_t s_lut[256];             /* pixel 0..255 -> int8 da quantize */

static tflite::MicroInterpreter *load(const unsigned char *data, int arena_kb, const char *name)
{
    static tflite::MicroMutableOpResolver<RESOLVER_SIZE> ops;
    static bool ops_ready;
    if (!ops_ready) { REGISTER_OPS(ops); ops_ready = true; }
    const tflite::Model *model = tflite::GetModel(data);
    if (model->version() != TFLITE_SCHEMA_VERSION) { ESP_LOGE(TAG, "%s: sai schema version", name); return nullptr; }
    uint8_t *arena = (uint8_t *)heap_caps_aligned_alloc(16, arena_kb * 1024, MALLOC_CAP_SPIRAM);
    if (!arena) { ESP_LOGE(TAG, "%s: khong cap phat duoc %d KB PSRAM", name, arena_kb); return nullptr; }
    auto *it = new tflite::MicroInterpreter(model, ops, arena, arena_kb * 1024);
    if (it->AllocateTensors() != kTfLiteOk) { ESP_LOGE(TAG, "%s: AllocateTensors loi (xem log op thieu)", name); return nullptr; }
    ESP_LOGI(TAG, "%s OK | arena dung %u / %d KB", name, (unsigned)it->arena_used_bytes() / 1024, arena_kb);
    return it;
}

int infer_init(void)
{
    s_det = load(DET_MODEL_DATA, DET_ARENA_KB, "detection");
    s_txt = load(TXT_MODEL_DATA, TXT_ARENA_KB, "textcnn");
    if (!s_det || !s_txt) { s_det = s_txt = nullptr; return -1; }

    TfLiteTensor *in = s_det->input(0), *out = s_det->output(0);
    if (in->type != kTfLiteInt8 || in->bytes != IMG_BYTES || out->dims->size != 4 ||
        out->dims->data[3] != 5 + NUM_CLASSES) {
        ESP_LOGE(TAG, "detection: can input int8 1x240x320x3, output NHWC ...x%d", 5 + NUM_CLASSES);
        s_det = s_txt = nullptr;
        return -1;
    }
    for (int v = 0; v < 256; v++)
        s_lut[v] = (int8_t)fminf(127, fmaxf(-128, roundf(v / 255.f / in->params.scale + in->params.zero_point)));

    if (s_txt->output(0)->dims->data[s_txt->output(0)->dims->size - 1] != NUM_INTENTS) {
        ESP_LOGE(TAG, "textcnn: output can %d intent", NUM_INTENTS);
        s_det = s_txt = nullptr;
        return -1;
    }
    ESP_LOGI(TAG, "vocab %d tu", VOCAB_LEN);
    return 0;
}

static float deq(const TfLiteTensor *t, int i)
{
    return t->type == kTfLiteInt8 ? (t->data.int8[i] - t->params.zero_point) * t->params.scale : t->data.f[i];
}

int infer_detect(const uint8_t *rgb, det_t *out, int max)
{
    if (!s_det) return -1;
    TfLiteTensor *in = s_det->input(0);
    for (int i = 0; i < IMG_BYTES; i++) in->data.int8[i] = s_lut[rgb[i]];
    if (s_det->Invoke() != kTfLiteOk) return -1;
    TfLiteTensor *o = s_det->output(0);
    int H = o->dims->data[1], W = o->dims->data[2], n = H * W * (5 + NUM_CLASSES);
    float *g = (float *)malloc(n * sizeof(float));
    if (!g) return -1;
    for (int i = 0; i < n; i++) g[i] = deq(o, i);
    int k = decode_grid(g, H, W, CONF_THRESH, NMS_IOU, out, max);
    free(g);
    return k;
}

int infer_intent(const char *norm_q)
{
    if (!s_txt) return -1;
    int32_t ids[SEQ_LEN];
    tokenize(norm_q, VOCAB_DATA, VOCAB_LEN, ids);
    TfLiteTensor *in = s_txt->input(0);
    for (int i = 0; i < SEQ_LEN; i++) {
        if (in->type == kTfLiteInt64) in->data.i64[i] = ids[i];
        else if (in->type == kTfLiteInt32) in->data.i32[i] = ids[i];
        else if (in->type == kTfLiteFloat32) in->data.f[i] = ids[i];
        else { ESP_LOGE(TAG, "textcnn: kieu input %d chua ho tro", in->type); return -1; }
    }
    if (s_txt->Invoke() != kTfLiteOk) return -1;
    TfLiteTensor *o = s_txt->output(0);
    int best = 0;
    for (int k = 1; k < NUM_INTENTS; k++) if (deq(o, k) > deq(o, best)) best = k;
    return best;
}
