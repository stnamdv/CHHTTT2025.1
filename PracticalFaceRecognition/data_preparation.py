import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import glob

class DataPreparation:
    def __init__(self, positive_dir="data/positive", negative_dir="data/negative", 
                 patch_size=(64, 64)):
        """
        Khởi tạo class chuẩn bị dữ liệu
        
        Args:
            positive_dir: Thư mục chứa ảnh có mặt
            negative_dir: Thư mục chứa ảnh không có mặt  
            patch_size: Kích thước patch (width, height)
        """
        self.positive_dir = positive_dir
        self.negative_dir = negative_dir
        self.patch_size = patch_size
        self.scaler = StandardScaler()
        
    def load_and_preprocess_images(self, image_paths, label):
        """
        Tải và tiền xử lý ảnh
        
        Args:
            image_paths: Danh sách đường dẫn ảnh
            label: Nhãn (1 cho positive, 0 cho negative)
            
        Returns:
            images: Mảng ảnh đã xử lý
            labels: Mảng nhãn tương ứng
        """
        images = []
        labels = []
        
        for path in image_paths:
            try:
                # Đọc ảnh
                img = cv2.imread(path)
                if img is None:
                    continue
                    
                # Chuyển sang grayscale
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Resize về kích thước patch
                img = cv2.resize(img, self.patch_size)
                
                # Chuẩn hóa pixel values về [0, 1]
                img = img.astype(np.float32) / 255.0
                
                # Flatten thành vector 1D
                img_vector = img.flatten()
                
                images.append(img_vector)
                labels.append(label)
                
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {path}: {e}")
                continue
                
        return np.array(images), np.array(labels)
    
    def generate_negative_samples(self, num_samples=1000):
        """
        Tạo negative samples từ ảnh không có mặt hoặc tự sinh
        
        Args:
            num_samples: Số lượng negative samples cần tạo
            
        Returns:
            negative_images: Mảng negative samples
            negative_labels: Nhãn 0 cho tất cả
        """
        negative_images = []
        negative_labels = []
        
        # Tạo random patches từ ảnh negative có sẵn
        negative_paths = glob.glob(os.path.join(self.negative_dir, "*.jpg")) + \
                        glob.glob(os.path.join(self.negative_dir, "*.png")) + \
                        glob.glob(os.path.join(self.negative_dir, "*.jpeg"))
        
        if not negative_paths:
            print(f"Không tìm thấy ảnh negative, tự sinh {num_samples} negative samples...")
            # Tạo negative samples đa dạng
            negative_images, negative_labels = self._generate_synthetic_negative_samples(num_samples)
        else:
            print(f"Tìm thấy {len(negative_paths)} ảnh negative, tạo {num_samples} patches...")
            # Kết hợp ảnh có sẵn và tự sinh
            synthetic_count = max(0, num_samples - len(negative_paths) * 10)  # Ước tính patches từ ảnh có sẵn
            
            # Tạo patches từ ảnh có sẵn
            for _ in range(num_samples - synthetic_count):
                img_path = np.random.choice(negative_paths)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                    
                # Tạo random patch từ ảnh
                h, w = img.shape
                if h > self.patch_size[1] and w > self.patch_size[0]:
                    y = np.random.randint(0, h - self.patch_size[1])
                    x = np.random.randint(0, w - self.patch_size[0])
                    patch = img[y:y+self.patch_size[1], x:x+self.patch_size[0]]
                    patch = patch.astype(np.float32) / 255.0
                    negative_images.append(patch.flatten())
                    negative_labels.append(0)
            
            # Bổ sung bằng synthetic samples nếu cần
            if synthetic_count > 0:
                print(f"Bổ sung {synthetic_count} synthetic negative samples...")
                synthetic_images, synthetic_labels = self._generate_synthetic_negative_samples(synthetic_count)
                negative_images.extend(synthetic_images)
                negative_labels.extend(synthetic_labels)
        
        return np.array(negative_images), np.array(negative_labels)
    
    def _generate_synthetic_negative_samples(self, num_samples):
        """
        Tạo negative samples tổng hợp đa dạng
        
        Args:
            num_samples: Số lượng samples cần tạo
            
        Returns:
            negative_images: Mảng negative samples
            negative_labels: Nhãn 0 cho tất cả
        """
        negative_images = []
        negative_labels = []
        
        for i in range(num_samples):
            # Tạo các loại pattern khác nhau
            pattern_type = i % 5  # 5 loại pattern khác nhau
            
            if pattern_type == 0:
                # Random noise
                patch = np.random.rand(*self.patch_size)
            elif pattern_type == 1:
                # Gradient patterns
                x, y = np.meshgrid(np.linspace(0, 1, self.patch_size[0]), 
                                 np.linspace(0, 1, self.patch_size[1]))
                patch = (x + y) / 2 + np.random.normal(0, 0.1, self.patch_size)
            elif pattern_type == 2:
                # Stripes patterns
                patch = np.zeros(self.patch_size)
                for j in range(0, self.patch_size[0], 8):
                    patch[:, j:j+4] = 0.8
            elif pattern_type == 3:
                # Checkerboard pattern
                patch = np.zeros(self.patch_size)
                for j in range(0, self.patch_size[0], 16):
                    for k in range(0, self.patch_size[1], 16):
                        if (j + k) % 32 == 0:
                            patch[j:j+16, k:k+16] = 0.7
            else:
                # Circular patterns
                center = (self.patch_size[0]//2, self.patch_size[1]//2)
                y, x = np.ogrid[:self.patch_size[1], :self.patch_size[0]]
                mask = (x - center[0])**2 + (y - center[1])**2 <= (min(self.patch_size)//3)**2
                patch = np.zeros(self.patch_size)
                patch[mask] = 0.6
                patch += np.random.normal(0, 0.1, self.patch_size)
            
            # Chuẩn hóa về [0, 1]
            patch = np.clip(patch, 0, 1)
            negative_images.append(patch.flatten())
            negative_labels.append(0)
        
        return negative_images, negative_labels
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Chuẩn bị dữ liệu huấn luyện và validation
        
        Args:
            test_size: Tỷ lệ dữ liệu validation
            random_state: Seed cho reproducibility
            
        Returns:
            X_train, X_val, y_train, y_val: Dữ liệu đã chia
        """
        print("Đang chuẩn bị dữ liệu...")
        
        # Tải positive samples
        positive_paths = glob.glob(os.path.join(self.positive_dir, "*.jpg")) + \
                        glob.glob(os.path.join(self.positive_dir, "*.png"))
        
        if not positive_paths:
            print("Không tìm thấy ảnh positive, tạo dữ liệu mẫu...")
            # Tạo dữ liệu mẫu cho demo
            positive_images, positive_labels = self._create_sample_data()
        else:
            positive_images, positive_labels = self.load_and_preprocess_images(positive_paths, 1)
        
        # Tạo negative samples - cân bằng với positive hoặc tối thiểu 1000
        target_negative_count = max(len(positive_images), 1000)
        print(f"Tạo {target_negative_count} negative samples để cân bằng với {len(positive_images)} positive samples...")
        negative_images, negative_labels = self.generate_negative_samples(
            num_samples=target_negative_count)
        
        # Kết hợp positive và negative
        X = np.vstack([positive_images, negative_images])
        y = np.hstack([positive_labels, negative_labels])
        
        print(f"Tổng số samples: {len(X)}")
        print(f"   - Positive samples: {np.sum(y == 1)}")
        print(f"   - Negative samples: {np.sum(y == 0)}")
        
        # Chuẩn hóa theo mean-variance
        X_scaled = self.scaler.fit_transform(X)
        
        # Chia train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y)
        
        print(f"Dữ liệu đã được chuẩn bị:")
        print(f"   - Train: {len(X_train)} samples")
        print(f"   - Validation: {len(X_val)} samples")
        
        return X_train, X_val, y_train, y_val
    
    def _create_sample_data(self):
        """
        Tạo dữ liệu mẫu cho demo (khi không có ảnh thật)
        """
        print("Tạo dữ liệu mẫu cho demo...")
        
        # Tạo positive samples (giả lập ảnh mặt)
        positive_samples = []
        for i in range(100):
            # Tạo pattern giống mặt người
            img = np.zeros(self.patch_size)
            
            # Vẽ hình tròn (đầu)
            center = (self.patch_size[0]//2, self.patch_size[1]//2)
            radius = min(self.patch_size) // 3
            cv2.circle(img, center, radius, 0.8, -1)
            
            # Vẽ mắt
            eye_y = center[1] - radius // 3
            cv2.circle(img, (center[0] - radius//2, eye_y), radius//6, 0.9, -1)
            cv2.circle(img, (center[0] + radius//2, eye_y), radius//6, 0.9, -1)
            
            # Vẽ mũi
            nose_y = center[1] + radius // 6
            cv2.circle(img, (center[0], nose_y), radius//8, 0.7, -1)
            
            # Thêm noise
            noise = np.random.normal(0, 0.1, img.shape)
            img = np.clip(img + noise, 0, 1)
            
            positive_samples.append(img.flatten())
        
        return np.array(positive_samples), np.ones(100)
    
    def visualize_samples(self, X, y, num_samples=8):
        """
        Trực quan hóa một số samples
        
        Args:
            X: Dữ liệu ảnh
            y: Nhãn
            num_samples: Số lượng samples hiển thị
        """
        fig, axes = plt.subplots(2, num_samples//2, figsize=(15, 6))
        axes = axes.flatten()
        
        # Lấy samples positive và negative
        positive_indices = np.where(y == 1)[0][:num_samples//2]
        negative_indices = np.where(y == 0)[0][:num_samples//2]
        
        for i, idx in enumerate(positive_indices):
            img = X[idx].reshape(self.patch_size)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Positive {i+1}')
            axes[i].axis('off')
        
        for i, idx in enumerate(negative_indices):
            img = X[idx].reshape(self.patch_size)
            axes[i + num_samples//2].imshow(img, cmap='gray')
            axes[i + num_samples//2].set_title(f'Negative {i+1}')
            axes[i + num_samples//2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def get_feature_dimension(self):
        """Trả về kích thước feature vector"""
        return self.patch_size[0] * self.patch_size[1]

if __name__ == "__main__":
    # Demo sử dụng
    data_prep = DataPreparation()
    X_train, X_val, y_train, y_val = data_prep.prepare_data()
    
    # Trực quan hóa samples
    data_prep.visualize_samples(X_train, y_train) 