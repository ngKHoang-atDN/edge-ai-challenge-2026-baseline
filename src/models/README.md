# Trọng số model v9b của đội (mảng C `.cc` + `.h`)

Thư mục này đang chứa model v9b của đội (không còn là baseline):
- `det_model.cc/.h`    ← `detector_J_int8.tflite` (177.531 tham số, int8)
- `textcnn_model.cc/.h` ← `intent_int8_v5.tflite` (TextCNN hash, 151 KB)
- `vocab.cc/.h` của baseline **không còn được dùng** (v9b mã hóa câu hỏi bằng hàm băm trong `main/logic.c`).

Sinh lại từ file `.tflite`:
```
python tools/tflite_to_cc.py ../detector_J_int8.tflite det_model main/models
python tools/tflite_to_cc.py ../intent_int8_v5.tflite textcnn_model main/models
```

Định dạng model v9b (khác baseline):
- Detection: input int8 `1x240x320x3`, **pixel 0..255 lượng tử hóa trực tiếp** (scale 1, zp -128, không chia 255).
  2 output int8: heatmap `1x60x80x7` và reg `1x60x80x4` (CenterNet stride 4) → `decode_v9b()` trong `main/logic.c`
  (không lọc đỉnh 3x3, NMS theo lớp IoU 0,55). 7 lớp: `bottle cup phone book mouse box toy_car`.
- TextCNN: input **int32 `[1,120]`** = 96 mã ký tự + 24 mã băm từ (FNV-1a % 2048 + 1) → `tokenize_v9b()`;
  output 5 intent `COUNT EXIST LIST LOCATION OUT_OF_SCOPE` (float32).

Kiểm tra trên máy (không cần board): xem `test/ref_v9b.py` (đối chiếu C với Python của notebook).
Arena, ngưỡng, danh sách op: `main/model_config.h`.
