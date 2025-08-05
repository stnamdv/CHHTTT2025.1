import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import maximum_filter
import time

class FaceDetector:
    def __init__(self, svm_trainer, patch_size=(64, 64), stride=8):
        """
        Khởi tạo Face Detector
        
        Args:
            svm_trainer: Đối tượng SVMTrainer đã được huấn luyện
            patch_size: Kích thước patch (width, height)
            stride: Bước nhảy của sliding window
        """
        self.svm_trainer = svm_trainer
        self.patch_size = patch_size
        self.stride = stride
        self.scaler = svm_trainer.scaler  # Sử dụng scaler từ training
        
    def extract_patches(self, image, stride=None):
        """
        Trích xuất patches từ ảnh sử dụng sliding window
        
        Args:
            image: Ảnh đầu vào (grayscale)
            stride: Bước nhảy (nếu None thì dùng self.stride)
            
        Returns:
            patches: Danh sách patches
            positions: Vị trí (x, y) của từng patch
        """
        if stride is None:
            stride = self.stride
            
        h, w = image.shape
        
        # Điều chỉnh stride cho ảnh lớn để không bỏ lỡ khuôn mặt
        if max(h, w) > 1000:
            stride = min(stride, 12)  # Giảm stride để phát hiện nhiều khuôn mặt hơn
            print(f"Ảnh lớn ({w}x{h}), giảm stride xuống {stride}")
        
        patches = []
        positions = []
        
        # Sliding window
        for y in range(0, h - self.patch_size[1] + 1, stride):
            for x in range(0, w - self.patch_size[0] + 1, stride):
                # Trích xuất patch
                patch = image[y:y + self.patch_size[1], x:x + self.patch_size[0]]
                
                # Chuẩn hóa
                patch = patch.astype(np.float32) / 255.0
                
                patches.append(patch.flatten())
                positions.append((x, y))
        
        return np.array(patches), positions
    
    def detect_faces(self, image_path, conf_thresh=0.5, conf_thresh_nms=0.3, 
                    nms_thresh=0.3, max_detections=5, visualize=True):
        """
        Phát hiện khuôn mặt trong ảnh
        
        Args:
            image_path: Đường dẫn ảnh
            conf_thresh: Ngưỡng confidence để lọc trước NMS
            conf_thresh_nms: Ngưỡng confidence sau NMS
            nms_thresh: Ngưỡng IoU cho NMS
            visualize: Có hiển thị kết quả không
            
        Returns:
            detections: Danh sách detections (x, y, w, h, confidence)
        """
        print(f"Đang phát hiện khuôn mặt trong {image_path}...")
        
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        
        # Chuyển sang grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Trích xuất patches
        patches, positions = self.extract_patches(gray)
        
        if len(patches) == 0:
            print("Không tìm thấy patches nào!")
            return []
        
        print(f"Đã trích xuất {len(patches)} patches")
        
        # Chuẩn hóa patches
        patches_scaled = self.scaler.transform(patches)
        
        # Dự đoán confidence scores
        confidence_scores = self.svm_trainer.predict_confidence(patches_scaled)
        
        # Điều chỉnh threshold cho ảnh lớn
        adjusted_conf_thresh = conf_thresh
        if max(image.shape[:2]) > 1000:
            # Giảm threshold cho ảnh có nhiều khuôn mặt
            adjusted_conf_thresh = max(conf_thresh - 0.1, 0.1)  # Giảm threshold
            print(f"Ảnh lớn, giảm confidence threshold: {conf_thresh} → {adjusted_conf_thresh}")
        
        # Lọc theo ngưỡng confidence
        high_conf_indices = np.where(confidence_scores > adjusted_conf_thresh)[0]
        
        if len(high_conf_indices) == 0:
            print("Không tìm thấy khuôn mặt nào!")
            return []
        
        print(f"Tìm thấy {len(high_conf_indices)} patches có confidence > {conf_thresh}")
        
        # Tạo detections
        detections = []
        for idx in high_conf_indices:
            x, y = positions[idx]
            confidence = confidence_scores[idx]
            detection = {
                'x': x,
                'y': y,
                'w': self.patch_size[0],
                'h': self.patch_size[1],
                'confidence': confidence
            }
            detections.append(detection)
        
        # Áp dụng Non-Maximum Suppression
        detections = self.non_maximum_suppression(detections, nms_thresh, conf_thresh_nms)
        
        # Lọc theo kích thước hợp lý
        detections = self.filter_by_size(detections, image.shape)
        
        # Giới hạn số lượng detections
        if len(detections) > max_detections:
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:max_detections]
            print(f"Giới hạn hiển thị {max_detections} detections có confidence cao nhất")
        
        print(f"Sau NMS và lọc: {len(detections)} khuôn mặt được phát hiện")
        
        # Trực quan hóa kết quả
        if visualize:
            self.visualize_detections(image, detections, confidence_scores, positions)
        
        return detections
    
    def non_maximum_suppression(self, detections, nms_thresh, conf_thresh_nms):
        """
        Áp dụng Non-Maximum Suppression để loại bỏ detections trùng lặp
        
        Args:
            detections: Danh sách detections
            nms_thresh: Ngưỡng IoU
            conf_thresh_nms: Ngưỡng confidence sau NMS
            
        Returns:
            filtered_detections: Danh sách detections sau NMS
        """
        if len(detections) == 0:
            return []
        
        # Sắp xếp theo confidence giảm dần
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        filtered_detections = []
        
        while detections:
            # Lấy detection có confidence cao nhất
            current = detections.pop(0)
            
            # Lọc theo ngưỡng confidence
            if current['confidence'] < conf_thresh_nms:
                continue
                
            filtered_detections.append(current)
            
            # Tính IoU với các detections còn lại
            remaining = []
            for detection in detections:
                iou = self.calculate_iou(current, detection)
                if iou < nms_thresh:
                    remaining.append(detection)
            
            detections = remaining
        
        return filtered_detections
    
    def calculate_iou(self, box1, box2):
        """
        Tính Intersection over Union (IoU) giữa 2 bounding boxes
        
        Args:
            box1, box2: Dictionaries chứa x, y, w, h
            
        Returns:
            iou: Giá trị IoU
        """
        # Tính tọa độ của intersection
        x1 = max(box1['x'], box2['x'])
        y1 = max(box1['y'], box2['y'])
        x2 = min(box1['x'] + box1['w'], box2['x'] + box2['w'])
        y2 = min(box1['y'] + box1['h'], box2['y'] + box2['h'])
        
        # Tính diện tích intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Tính diện tích union
        area1 = box1['w'] * box1['h']
        area2 = box2['w'] * box2['h']
        union = area1 + area2 - intersection
        
        return intersection / union
    
    def filter_by_size(self, detections, image_shape):
        """
        Lọc detections theo kích thước hợp lý
        
        Args:
            detections: Danh sách detections
            image_shape: Kích thước ảnh (h, w, c)
            
        Returns:
            filtered_detections: Detections sau khi lọc
        """
        if len(detections) == 0:
            return detections
        
        h, w = image_shape[:2]
        
        # Điều chỉnh theo kích thước ảnh
        if max(h, w) > 1000:  # Ảnh lớn
            min_face_size = min(h, w) * 0.02  # Giảm xuống 2% để phát hiện khuôn mặt nhỏ
            max_face_size = min(h, w) * 0.6   # Tăng lên 60% để phát hiện khuôn mặt lớn
        else:  # Ảnh nhỏ
            min_face_size = min(h, w) * 0.1   # 10%
            max_face_size = min(h, w) * 0.8   # 80%
        
        filtered_detections = []
        for detection in detections:
            face_size = min(detection['w'], detection['h'])
            if min_face_size <= face_size <= max_face_size:
                filtered_detections.append(detection)
        
        if len(filtered_detections) != len(detections):
            print(f"Lọc kích thước: {len(detections)} → {len(filtered_detections)} detections")
            print(f"  - Kích thước hợp lý: {min_face_size:.1f} - {max_face_size:.1f} pixels")
        
        return filtered_detections
    
    def visualize_detections(self, image, detections, confidence_scores, positions):
        """
        Trực quan hóa kết quả phát hiện
        
        Args:
            image: Ảnh gốc
            detections: Danh sách detections
            confidence_scores: Tất cả confidence scores
            positions: Tất cả vị trí patches
        """
        # Tạo heatmap
        h, w = image.shape[:2]
        heatmap = np.zeros((h, w))
        
        # Vẽ confidence scores lên heatmap
        for i, (x, y) in enumerate(positions):
            if confidence_scores[i] > 0:
                heatmap[y:y+self.patch_size[1], x:x+self.patch_size[0]] += confidence_scores[i]
        
        # Chuẩn hóa heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Vẽ kết quả
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Ảnh gốc với bounding boxes
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for detection in detections:
            x, y, w, h = detection['x'], detection['y'], detection['w'], detection['h']
            confidence = detection['confidence']
            
            # Vẽ bounding box
            rect = plt.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
            
            # Vẽ confidence score
            axes[0].text(x, y-5, f'{confidence:.3f}', 
                        color='red', fontsize=10, weight='bold')
        
        axes[0].set_title('Face Detections')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='hot')
        axes[1].set_title('Confidence Heatmap')
        axes[1].axis('off')
        
        # Histogram confidence scores
        axes[2].hist(confidence_scores, bins=50, alpha=0.7, color='blue')
        axes[2].axvline(x=0, color='red', linestyle='--', label='Decision boundary')
        axes[2].set_xlabel('Confidence Score')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Confidence Score Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Đã trực quan hóa {len(detections)} khuôn mặt được phát hiện")
    
    def detect_faces_multi_scale(self, image_path, scales=[0.3, 0.5, 0.7, 1.0, 1.3, 1.6], 
                                conf_thresh=0.5, conf_thresh_nms=0.3, nms_thresh=0.3,
                                max_detections=20):
        """
        Phát hiện khuôn mặt với nhiều scale khác nhau
        
        Args:
            image_path: Đường dẫn ảnh
            scales: Danh sách các scale để thử
            conf_thresh: Ngưỡng confidence
            conf_thresh_nms: Ngưỡng confidence sau NMS
            nms_thresh: Ngưỡng IoU cho NMS
            
        Returns:
            all_detections: Tất cả detections từ các scale
        """
        print(f"Phát hiện khuôn mặt multi-scale...")
        
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        
        all_detections = []
        
        for scale in scales:
            print(f"Scale: {scale}x")
            
            # Resize ảnh
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(image, (new_w, new_h))
            
            # Chuyển sang grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Trích xuất patches
            patches, positions = self.extract_patches(gray)
            
            if len(patches) == 0:
                continue
            
            # Chuẩn hóa và dự đoán
            patches_scaled = self.scaler.transform(patches)
            confidence_scores = self.svm_trainer.predict_confidence(patches_scaled)
            
            # Lọc theo ngưỡng confidence
            high_conf_indices = np.where(confidence_scores > conf_thresh)[0]
            
            # Tạo detections với scale factor
            for idx in high_conf_indices:
                x, y = positions[idx]
                confidence = confidence_scores[idx]
                
                # Scale về kích thước gốc
                x_orig = int(x / scale)
                y_orig = int(y / scale)
                w_orig = int(self.patch_size[0] / scale)
                h_orig = int(self.patch_size[1] / scale)
                
                detection = {
                    'x': x_orig,
                    'y': y_orig,
                    'w': w_orig,
                    'h': h_orig,
                    'confidence': confidence,
                    'scale': scale
                }
                all_detections.append(detection)
        
        # Áp dụng NMS trên tất cả detections
        final_detections = self.non_maximum_suppression(all_detections, nms_thresh, conf_thresh_nms)
        
        # Lọc theo kích thước
        final_detections = self.filter_by_size(final_detections, image.shape)
        
        # Giới hạn số lượng detections
        if len(final_detections) > max_detections:
            final_detections = sorted(final_detections, key=lambda x: x['confidence'], reverse=True)[:max_detections]
            print(f"Giới hạn multi-scale: {max_detections} detections có confidence cao nhất")
        
        print(f"Multi-scale detection: {len(final_detections)} khuôn mặt")
        
        # Trực quan hóa
        self.visualize_multi_scale_detections(image, final_detections)
        
        return final_detections
    
    def visualize_multi_scale_detections(self, image, detections):
        """
        Trực quan hóa kết quả multi-scale detection
        """
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Vẽ bounding boxes với màu khác nhau cho từng scale
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for detection in detections:
            x, y, w, h = detection['x'], detection['y'], detection['w'], detection['h']
            confidence = detection['confidence']
            scale = detection['scale']
            
            # Chọn màu dựa trên scale
            color_idx = min(int(scale * 2), len(colors) - 1)
            color = colors[color_idx]
            
            # Vẽ bounding box
            rect = plt.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)
            
            # Vẽ label
            plt.text(x, y-5, f'{confidence:.3f} ({scale:.1f}x)', 
                    color=color, fontsize=8, weight='bold')
        
        plt.title('Multi-Scale Face Detection')
        plt.axis('off')
        plt.show()



if __name__ == "__main__":
    # Demo sử dụng
    from data_preparation import DataPreparation
    from svm_training import SVMTrainer
    
    # Chuẩn bị dữ liệu và huấn luyện SVM
    print("Khởi tạo hệ thống phát hiện khuôn mặt...")
    
    data_prep = DataPreparation()
    X_train, X_val, y_train, y_val = data_prep.prepare_data()
    
    svm_trainer = SVMTrainer()
    best_C, best_acc = svm_trainer.find_best_C(X_train, y_train, X_val, y_val)
    
    # Tạo ảnh test
    test_image_path = create_test_image()
    
    # Phát hiện khuôn mặt
    detector = FaceDetector(svm_trainer)
    
    # Single scale detection
    detections = detector.detect_faces(test_image_path, conf_thresh=0.0, conf_thresh_nms=0.0)
    
    # Multi-scale detection
    multi_detections = detector.detect_faces_multi_scale(test_image_path, 
                                                       scales=[0.5, 1.0, 1.5], 
                                                       conf_thresh=0.0, conf_thresh_nms=0.0) 