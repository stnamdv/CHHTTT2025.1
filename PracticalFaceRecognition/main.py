#!/usr/bin/env python3
"""
Hệ thống phát hiện khuôn mặt sử dụng SVM
Quy trình 3 phần:
1. Chuẩn bị dữ liệu huấn luyện
2. Huấn luyện và đánh giá SVM  
3. Phát hiện khuôn mặt trong ảnh
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
from data_preparation import DataPreparation
from svm_training import SVMTrainer
from face_detection import FaceDetector

def main():
    """
    Hàm chính chạy toàn bộ hệ thống phát hiện khuôn mặt
    """
    parser = argparse.ArgumentParser(description='Hệ thống phát hiện khuôn mặt SVM')
    parser.add_argument('--mode', choices=['full', 'data', 'train', 'detect'], 
                       default='full', help='Chế độ chạy')
    parser.add_argument('--positive_dir', default='data/positive', 
                       help='Thư mục ảnh positive')
    parser.add_argument('--negative_dir', default='data/negative', 
                       help='Thư mục ảnh negative')
    parser.add_argument('--test_image', default=None, 
                       help='Đường dẫn ảnh test')
    parser.add_argument('--patch_size', type=int, nargs=2, default=[64, 64],
                       help='Kích thước patch (width height)')
    parser.add_argument('--stride', type=int, default=8,
                       help='Bước nhảy sliding window')
    parser.add_argument('--conf_thresh', type=float, default=0.8,
                       help='Ngưỡng confidence (tăng để loại noise)')
    parser.add_argument('--conf_thresh_nms', type=float, default=0.5,
                       help='Ngưỡng confidence sau NMS')
    parser.add_argument('--nms_thresh', type=float, default=0.5,
                       help='Ngưỡng IoU cho NMS (tăng để loại box trùng)')
    parser.add_argument('--max_detections', type=int, default=15,
                       help='Số lượng detections tối đa hiển thị')
    
    args = parser.parse_args()
    
    print("HỆ THỐNG PHÁT HIỆN KHUÔN MẶT SVM")
    print("=" * 50)
    
    # Khởi tạo các đối tượng
    data_prep = DataPreparation(
        positive_dir=args.positive_dir,
        negative_dir=args.negative_dir,
        patch_size=tuple(args.patch_size)
    )
    
    svm_trainer = SVMTrainer(patch_size=tuple(args.patch_size))
    
    # ========================================
    # PART 1: CHUẨN BỊ DỮ LIỆU HUẤN LUYỆN
    # ========================================
    if args.mode in ['full', 'data']:
        print("\nPART 1: CHUẨN BỊ DỮ LIỆU HUẤN LUYỆN")
        print("-" * 40)
        
        # Chuẩn bị dữ liệu
        X_train, X_val, y_train, y_val = data_prep.prepare_data()
        
        # Truyền scaler cho SVM trainer
        svm_trainer.scaler = data_prep.scaler
        
        # Trực quan hóa samples
        print("\nTrực quan hóa samples...")
        data_prep.visualize_samples(X_train, y_train)
        
        print(f"Part 1 hoàn thành!")
        print(f"   - Train samples: {len(X_train)}")
        print(f"   - Validation samples: {len(X_val)}")
        print(f"   - Feature dimension: {data_prep.get_feature_dimension()}")
    
    # ========================================
    # PART 2: HUẤN LUYỆN VÀ ĐÁNH GIÁ SVM
    # ========================================
    if args.mode in ['full', 'train']:
        print("\nPART 2: HUẤN LUYỆN VÀ ĐÁNH GIÁ SVM")
        print("-" * 40)
        
        # Chuẩn bị dữ liệu nếu chưa có
        if args.mode == 'train':
            X_train, X_val, y_train, y_val = data_prep.prepare_data()
            # Truyền scaler cho SVM trainer
            svm_trainer.scaler = data_prep.scaler
        
        # Sử dụng C=0.1 (ổn định nhất theo phân tích)
        print("\nSử dụng C=0.1 (ổn định nhất)...")
        best_C = 0.1
        svm_trainer.train_svm(X_train, y_train, C=best_C)
        best_acc = svm_trainer.evaluate_svm(X_val, y_val)
        svm_trainer.compute_hyperplane()
        
        # Phân tích hiệu ứng regularization
        print("\nPhân tích hiệu ứng regularization...")
        svm_trainer.analyze_regularization_effect(X_train, y_train, X_val, y_val)
        
        # Trực quan hóa weight vector
        print("\nTrực quan hóa weight vector...")
        svm_trainer.visualize_weight_vector("Weight Vector W (Best Model)")
        
        print(f"Part 2 hoàn thành!")
        print(f"   - Best C: {best_C}")
        print(f"   - Best accuracy: {best_acc:.4f}")
        print(f"   - Support vectors: {len(svm_trainer.svm.support_vectors_)}")
    
    # ========================================
    # PART 3: PHÁT HIỆN KHUÔN MẶT TRONG ẢNH
    # ========================================
    if args.mode in ['full', 'detect']:
        print("\nPART 3: PHÁT HIỆN KHUÔN MẶT TRONG ẢNH")
        print("-" * 40)
        
        # Chuẩn bị dữ liệu và huấn luyện SVM nếu chưa có
        if args.mode == 'detect':
            X_train, X_val, y_train, y_val = data_prep.prepare_data()
            # Truyền scaler cho SVM trainer
            svm_trainer.scaler = data_prep.scaler
            # Sử dụng C=0.1 (ổn định nhất)
            best_C = 0.1
            svm_trainer.train_svm(X_train, y_train, C=best_C)
            best_acc = svm_trainer.evaluate_svm(X_val, y_val)
            svm_trainer.compute_hyperplane()
        
        # Sử dụng ảnh test mặc định hoặc ảnh được chỉ định
        if args.test_image is None:
            # Sử dụng test_image.jpg có sẵn
            test_image_path = 'test_image.jpg'
            if not os.path.exists(test_image_path):
                raise FileNotFoundError(f"Không tìm thấy file {test_image_path}. Vui lòng đặt file ảnh test vào thư mục gốc.")
            print(f"\nSử dụng ảnh test có sẵn: {test_image_path}")
        else:
            test_image_path = args.test_image
            print(f"\nSử dụng ảnh test được chỉ định: {test_image_path}")
        
        # Khởi tạo face detector
        detector = FaceDetector(
            svm_trainer=svm_trainer,
            patch_size=tuple(args.patch_size),
            stride=args.stride
        )
        
        # Single scale detection
        print(f"\nSingle scale detection...")
        detections = detector.detect_faces(
            test_image_path,
            conf_thresh=args.conf_thresh,
            conf_thresh_nms=args.conf_thresh_nms,
            nms_thresh=args.nms_thresh,
            max_detections=args.max_detections
        )
        
        # Multi-scale detection
        print(f"\nMulti-scale detection...")
        multi_detections = detector.detect_faces_multi_scale(
            test_image_path,
            scales=[0.3, 0.5, 0.7, 1.0, 1.3, 1.6],
            conf_thresh=args.conf_thresh,
            conf_thresh_nms=args.conf_thresh_nms,
            nms_thresh=args.nms_thresh,
            max_detections=args.max_detections
        )
        
        print(f"Part 3 hoàn thành!")
        print(f"   - Single scale detections: {len(detections)}")
        print(f"   - Multi-scale detections: {len(multi_detections)}")
        print(f"   - Test image: {test_image_path}")
    
    # ========================================
    # TỔNG KẾT
    # ========================================
    print("\nTỔNG KẾT HỆ THỐNG")
    print("=" * 50)
    
    if args.mode in ['full', 'data']:
        print(f"Part 1 - Data Preparation:")
        print(f"   - Feature dimension: {data_prep.get_feature_dimension()}")
        print(f"   - Data normalization: Mean-Variance")
    
    if args.mode in ['full', 'train']:
        print(f"Part 2 - SVM Training:")
        print(f"   - Best C: {svm_trainer.best_C}")
        print(f"   - Best accuracy: {svm_trainer.best_accuracy:.4f}")
        print(f"   - Weight vector norm: {np.linalg.norm(svm_trainer.W):.6f}")
    
    if args.mode in ['full', 'detect']:
        print(f"Part 3 - Face Detection:")
        print(f"   - Patch size: {args.patch_size}")
        print(f"   - Stride: {args.stride}")
        print(f"   - Confidence threshold: {args.conf_thresh}")
        print(f"   - NMS threshold: {args.nms_thresh}")
    
    print(f"\nHệ thống phát hiện khuôn mặt đã hoàn thành!")
    print(f"Gợi ý:")
    print(f"   - Với C nhỏ, W trông giống mặt hơn (regularization mạnh)")
    print(f"   - Với C lớn, W có thể overfit (regularization yếu)")
    print(f"   - NMS giúp loại bỏ detections trùng lặp")
    print(f"   - Multi-scale detection cải thiện độ chính xác")

def demo_quick():
    """
    Demo nhanh cho người dùng mới
    """
    print("DEMO NHANH HỆ THỐNG PHÁT HIỆN KHUÔN MẶT")
    print("=" * 50)
    
    try:
        # Import các module cần thiết
        import numpy as np
        from data_preparation import DataPreparation
        from svm_training import SVMTrainer
        from face_detection import FaceDetector
        
        print("Đang khởi tạo hệ thống...")
        
        # Part 1: Chuẩn bị dữ liệu
        print("\nPart 1: Chuẩn bị dữ liệu...")
        data_prep = DataPreparation()
        X_train, X_val, y_train, y_val = data_prep.prepare_data()
        
        # Part 2: Huấn luyện SVM
        print("\nPart 2: Huấn luyện SVM...")
        svm_trainer = SVMTrainer()
        # Truyền scaler cho SVM trainer
        svm_trainer.scaler = data_prep.scaler
        # Sử dụng C=0.1 (ổn định nhất)
        best_C = 0.1
        svm_trainer.train_svm(X_train, y_train, C=best_C)
        best_acc = svm_trainer.evaluate_svm(X_val, y_val)
        svm_trainer.compute_hyperplane()
        
        # Part 3: Phát hiện khuôn mặt
        print("\nPart 3: Phát hiện khuôn mặt...")
        
        # Sử dụng ảnh test có sẵn
        test_image_path = 'test_image.jpg'
        if not os.path.exists(test_image_path):
            raise FileNotFoundError(f"Không tìm thấy file {test_image_path}. Vui lòng đặt file ảnh test vào thư mục gốc.")
        print(f"Sử dụng ảnh test có sẵn: {test_image_path}")
            
        detector = FaceDetector(svm_trainer)
        detections = detector.detect_faces(test_image_path, conf_thresh=0.0, conf_thresh_nms=0.0)
        
        print(f"\nDemo hoàn thành")
        print(f"   - Accuracy: {best_acc:.4f}")
        print(f"   - Detections: {len(detections)}")
        
    except Exception as e:
        print(f"Lỗi trong demo: {e}")
        print("Hãy chạy: python main.py --mode full")

if __name__ == "__main__":
    # Import numpy cho main
    import numpy as np
    
    if len(sys.argv) == 1:
        # Nếu không có argument, chạy demo nhanh
        demo_quick()
    else:
        # Chạy với arguments
        main() 