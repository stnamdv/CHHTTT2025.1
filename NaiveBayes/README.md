# Bài Tập Naive Bayes Classification

## Mô tả bài tập
Dựa trên bộ dữ liệu huấn luyện về việc mua máy tính, sử dụng thuật toán Naive Bayes để dự đoán lớp của một ví dụ mới.

## Dữ liệu huấn luyện
Bộ dữ liệu gồm 14 mẫu với các thuộc tính:
- **age**: tuổi (<=30, 31...40, >40)
- **income**: thu nhập (high, medium, low)
- **student**: có phải sinh viên không (yes, no)
- **credit_rating**: xếp hạng tín dụng (fair, excellent)
- **Class**: có mua máy tính không (yes, no)

## Ví dụ cần dự đoán
- **age**: <=30
- **income**: medium
- **student**: yes
- **credit_rating**: fair

## Cách chạy
1. Cài đặt thư viện cần thiết: `pip install pandas numpy scikit-learn`
2. Chạy file `naive_bayes_exercise.py`
3. Xem kết quả dự đoán và các bước tính toán chi tiết

## Kết quả mong đợi
Chương trình sẽ hiển thị:
- Bảng dữ liệu huấn luyện
- Xác suất tiên nghiệm P(C)
- Xác suất có điều kiện P(X|C) cho từng thuộc tính
- Xác suất hậu nghiệm P(C|X)
- Kết quả dự đoán cuối cùng 