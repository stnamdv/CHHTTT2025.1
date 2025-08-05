# Bài Tập Decision Tree Classifier

## Mô tả bài tập

Bài tập này thực hiện việc xây dựng một Decision Tree Classifier để dự đoán rủi ro của tài xế dựa trên các thuộc tính sau:

- **time**: Thời gian có bằng lái xe (1-2 năm, 2-7 năm, >7 năm)
- **gender**: Giới tính (nam, nữ)
- **area**: Khu vực sinh sống (thành thị, nông thôn)
- **risk**: Lớp rủi ro (thấp, cao) - đây là biến mục tiêu

## Dữ liệu huấn luyện

| ID  | time | gender | area  | risk |
| --- | ---- | ------ | ----- | ---- |
| 1   | 1-2  | m      | urban | low  |
| 2   | 2-7  | m      | rural | high |
| 3   | >7   | f      | rural | low  |
| 4   | 1-2  | f      | rural | high |
| 5   | >7   | m      | rural | high |
| 6   | 1-2  | m      | rural | high |
| 7   | 2-7  | f      | urban | low  |
| 8   | 2-7  | m      | urban | low  |

## Dữ liệu kiểm tra

| ID  | time | gender | area  |
| --- | ---- | ------ | ----- |
| A   | 1-2  | f      | rural |
| B   | 2-7  | m      | urban |
| C   | 1-2  | f      | urban |

## Cài đặt

1. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

2. Chạy chương trình:

```bash
python decision_tree_classifier.py
```

## Các tính năng của chương trình

1. **Tạo dữ liệu**: Tự động tạo dữ liệu huấn luyện và kiểm tra từ bài tập
2. **Tiền xử lý**: Mã hóa các biến categorical thành số
3. **Xây dựng cây**: Sử dụng thuật toán ID3 với entropy
4. **Trực quan hóa**: Vẽ decision tree và lưu thành file PNG
5. **Phân tích cấu trúc**: Hiển thị chi tiết cấu trúc của cây
6. **Dự đoán**: Dự đoán rủi ro cho dữ liệu kiểm tra
7. **Tính toán thủ công**: Tính entropy và information gain thủ công

## Kết quả mong đợi

Chương trình sẽ:

- Hiển thị decision tree được xây dựng
- Dự đoán rủi ro cho 3 trường hợp A, B, C
- Tính toán và hiển thị entropy, information gain
- Lưu hình ảnh decision tree vào file `decision_tree.png`

## Giải thích thuật toán

### Entropy

Entropy đo lường độ không chắc chắn trong dữ liệu:

```
Entropy(S) = -Σ(p_i * log2(p_i))
```

### Information Gain

Information Gain đo lường mức độ giảm entropy khi phân chia theo một thuộc tính:

```
IG(S, A) = Entropy(S) - Σ(|S_v|/|S| * Entropy(S_v))
```

### Quy trình xây dựng cây

1. Tính entropy của tập dữ liệu gốc
2. Với mỗi thuộc tính, tính information gain
3. Chọn thuộc tính có information gain cao nhất làm node gốc
4. Lặp lại quá trình cho các node con
5. Dừng khi tất cả mẫu trong node thuộc cùng một lớp
