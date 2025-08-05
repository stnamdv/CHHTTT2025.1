# Hệ thống phát hiện khuôn mặt sử dụng SVM

Hệ thống phát hiện khuôn mặt hoàn chỉnh sử dụng Support Vector Machine (SVM) với quy trình 3 phần rõ ràng.

## Tổng quan

Hệ thống này triển khai phương pháp phát hiện khuôn mặt cổ điển sử dụng SVM tuyến tính với sliding window và Non-Maximum Suppression (NMS).

### Quy trình 3 phần:

1. **Part 1: Chuẩn bị dữ liệu huấn luyện**

   - Tải và trực quan hóa ảnh huấn luyện (positive/negative)
   - Chuẩn hóa theo mean-variance
   - Định dạng dữ liệu cho SVM

2. **Part 2: Huấn luyện và đánh giá SVM**

   - Huấn luyện SVM tuyến tính
   - Tính toán hyperplane W từ support vectors
   - Phân tích hiệu ứng regularization
   - Trực quan hóa weight vector

3. **Part 3: Phát hiện khuôn mặt trong ảnh**
   - Sliding window detection
   - Confidence scoring
   - Non-Maximum Suppression (NMS)
   - Multi-scale detection

## Cài đặt

### Yêu cầu hệ thống

- Python 3.7+
- OpenCV
- scikit-learn
- matplotlib
- numpy
- scipy

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## Cấu trúc dự án

```
PracticalFaceRecognition/
├── data_preparation.py    # Part 1: Chuẩn bị dữ liệu
├── svm_training.py        # Part 2: Huấn luyện SVM
├── face_detection.py      # Part 3: Phát hiện khuôn mặt
├── main.py               # File chính chạy toàn bộ hệ thống
├── requirements.txt      # Dependencies
└── README.md            # Hướng dẫn này
```

## Sử dụng

### Demo nhanh

```bash
python main.py
```

### Chạy toàn bộ hệ thống

```bash
python main.py --mode full
```

### Chạy từng phần riêng biệt

#### Part 1: Chuẩn bị dữ liệu

```bash
python main.py --mode data
```

#### Part 2: Huấn luyện SVM

```bash
python main.py --mode train
```

#### Part 3: Phát hiện khuôn mặt

```bash
python main.py --mode detect
```

### Tùy chỉnh tham số

```bash
python main.py --mode full \
    --patch_size 64 64 \
    --stride 8 \
    --conf_thresh 0.5 \
    --conf_thresh_nms 0.3 \
    --nms_thresh 0.3
```

## 📊 Các tham số quan trọng

### Data Preparation

- `patch_size`: Kích thước patch (mặc định: 64x64)
- `positive_dir`: Thư mục ảnh có mặt
- `negative_dir`: Thư mục ảnh không có mặt

### SVM Training

- `C`: Tham số regularization (mặc định: tự động tìm)
- `kernel`: Loại kernel (mặc định: linear)

### Face Detection

- `stride`: Bước nhảy sliding window (mặc định: 8)
- `conf_thresh`: Ngưỡng confidence trước NMS (mặc định: 0.5)
- `conf_thresh_nms`: Ngưỡng confidence sau NMS (mặc định: 0.3)
- `nms_thresh`: Ngưỡng IoU cho NMS (mặc định: 0.3)

## 🔍 Phân tích kỹ thuật

### Part 1: Chuẩn bị dữ liệu

```python
from data_preparation import DataPreparation

# Khởi tạo
data_prep = DataPreparation(patch_size=(64, 64))

# Chuẩn bị dữ liệu
X_train, X_val, y_train, y_val = data_prep.prepare_data()

# Trực quan hóa
data_prep.visualize_samples(X_train, y_train)
```

**Tính năng:**

- Tự động tạo dữ liệu mẫu nếu không có ảnh thật
- Chuẩn hóa mean-variance
- Trực quan hóa positive/negative samples

### Part 2: Huấn luyện SVM

```python
from svm_training import SVMTrainer

# Khởi tạo
svm_trainer = SVMTrainer()

# Tìm C tốt nhấtpip install -r requirements.txt
best_C, best_acc = svm_trainer.find_best_C(X_train, y_train, X_val, y_val)

# Phân tích regularization
svm_trainer.analyze_regularization_effect(X_train, y_train, X_val, y_val)

# Trực quan hóa weight vector
svm_trainer.visualize_weight_vector()
```

**Tính năng:**

- Tự động tìm giá trị C tối ưu
- Tính toán hyperplane W từ support vectors
- Phân tích hiệu ứng regularization
- Trực quan hóa weight vector (giống mặt người)

### Part 3: Phát hiện khuôn mặt

```python
from face_detection import FaceDetector

# Khởi tạo detector
detector = FaceDetector(svm_trainer)

# Single scale detection
detections = detector.detect_faces("test_image.jpg")

# Multi-scale detection
multi_detections = detector.detect_faces_multi_scale(
    "test_image.jpg",
    scales=[0.5, 1.0, 1.5, 2.0]
)
```

**Tính năng:**

- Sliding window với stride tùy chỉnh
- Confidence scoring từ hyperplane
- Non-Maximum Suppression (NMS)
- Multi-scale detection
- Trực quan hóa heatmap và detections

## 🎨 Trực quan hóa

Hệ thống cung cấp nhiều loại trực quan hóa:

1. **Samples visualization**: Positive/negative samples
2. **Weight vector visualization**: W như ảnh (giống mặt người)
3. **Regularization analysis**: Accuracy vs C, ||W|| vs C
4. **Detection results**: Bounding boxes, heatmap, confidence distribution
5. **Multi-scale detection**: Detections từ nhiều scale

## 🔬 Phân tích regularization

**Gợi ý phân tích:**

- **C nhỏ** → ||W|| nhỏ → Regularization mạnh → W giống mặt hơn
- **C lớn** → ||W|| lớn → Regularization yếu → W có thể overfit

Khi C nhỏ, mô hình được regularize mạnh, làm cho weight vector W gần với trung bình của ảnh mặt người hơn.

## 📈 Kết quả mẫu

### Accuracy vs Regularization Parameter C

```
C = 0.1  → Accuracy: 0.85
C = 1.0  → Accuracy: 0.92  ← Best
C = 10.0 → Accuracy: 0.89
```

### Weight Vector Analysis

```
||W|| (C=0.1)  = 0.45  ← Regularization mạnh
||W|| (C=1.0)  = 0.78  ← Cân bằng
||W|| (C=10.0) = 1.23  ← Regularization yếu
```

## 🛠️ Tùy chỉnh nâng cao

### Thêm dữ liệu thật

```bash
# Tạo thư mục dữ liệu
mkdir -p data/positive data/negative

# Thêm ảnh có mặt vào data/positive/
# Thêm ảnh không có mặt vào data/negative/
```

### Tùy chỉnh hyperparameters

```python
# Trong main.py
svm_trainer.find_best_C(X_train, y_train, X_val, y_val,
                       C_values=[0.01, 0.1, 1.0, 10.0, 100.0])
```

### Tùy chỉnh detection parameters

```python
detector.detect_faces(
    image_path,
    conf_thresh=0.3,      # Ngưỡng thấp hơn = nhiều detections
    conf_thresh_nms=0.1,  # Ngưỡng thấp hơn = ít lọc
    nms_thresh=0.5        # Ngưỡng cao hơn = ít NMS
)
```

## 🐛 Troubleshooting

### Lỗi thường gặp

1. **ImportError**: Cài đặt dependencies

   ```bash
   pip install -r requirements.txt
   ```

2. **MemoryError**: Giảm patch_size hoặc stride

   ```bash
   python main.py --patch_size 32 32 --stride 16
   ```

3. **No detections**: Giảm confidence threshold
   ```bash
   python main.py --conf_thresh 0.0 --conf_thresh_nms 0.0
   ```

### Performance tips

- **Tăng tốc**: Giảm stride hoặc patch_size
- **Độ chính xác**: Tăng số scale trong multi-scale detection
- **Memory**: Giảm số lượng patches hoặc sử dụng batch processing

## 📚 Tài liệu tham khảo

- [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
- [Face Detection with SVM](https://en.wikipedia.org/wiki/Face_detection)
- [Non-Maximum Suppression](https://en.wikipedia.org/wiki/Non-maximum_suppression)
- [Sliding Window](https://en.wikipedia.org/wiki/Sliding_window_protocol)

## 🤝 Đóng góp

Hệ thống này được thiết kế để học tập và nghiên cứu. Mọi đóng góp đều được chào đón!

## 📄 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

---

**🎉 Chúc bạn thành công với hệ thống phát hiện khuôn mặt!**
