import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import cv2

class SVMTrainer:
    def __init__(self, patch_size=(64, 64)):
        """
        Khởi tạo SVM Trainer
        
        Args:
            patch_size: Kích thước patch (width, height)
        """
        self.patch_size = patch_size
        self.svm = None
        self.best_C = None
        self.best_accuracy = 0
        self.W = None
        self.b = None
        self.scaler = None  # Sẽ được set từ DataPreparation
        
    def train_svm(self, X_train, y_train, C=1.0, kernel='linear'):
        """
        Huấn luyện SVM
        
        Args:
            X_train: Dữ liệu huấn luyện
            y_train: Nhãn huấn luyện
            C: Tham số regularization
            kernel: Loại kernel ('linear', 'rbf', etc.)
            
        Returns:
            Trained SVM model
        """
        print(f"🚀 Huấn luyện SVM với C={C}, kernel={kernel}...")
        
        # Khởi tạo SVM
        self.svm = SVC(C=C, kernel=kernel, random_state=42)
        
        # Huấn luyện
        self.svm.fit(X_train, y_train)
        
        print(f"✅ SVM đã được huấn luyện!")
        print(f"   - Số support vectors: {len(self.svm.support_vectors_)}")
        print(f"   - Support vector indices: {len(self.svm.support_)}")
        
        return self.svm
    
    def evaluate_svm(self, X_val, y_val):
        """
        Đánh giá SVM trên tập validation
        
        Args:
            X_val: Dữ liệu validation
            y_val: Nhãn validation
            
        Returns:
            accuracy: Độ chính xác
        """
        if self.svm is None:
            raise ValueError("SVM chưa được huấn luyện!")
        
        # Dự đoán
        y_pred = self.svm.predict(X_val)
        
        # Tính accuracy
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"📊 Kết quả đánh giá:")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - Classification Report:")
        print(classification_report(y_val, y_pred))
        
        return accuracy
    
    def compute_hyperplane(self):
        """
        Tính toán hyperplane W từ support vectors và alpha coefficients
        
        Returns:
            W: Weight vector của hyperplane
            b: Bias term
        """
        if self.svm is None:
            raise ValueError("SVM chưa được huấn luyện!")
        
        if self.svm.kernel != 'linear':
            print("⚠️ Chỉ có thể tính W cho linear kernel!")
            return None, None
        
        # Cách đơn giản hơn: sử dụng coef_ và intercept_ có sẵn
        W = self.svm.coef_[0]  # Weight vector
        b = self.svm.intercept_[0]  # Bias term
        
        self.W = W
        self.b = b
        
        print(f"Hyperplane đã được tính toán:")
        print(f"   - W shape: {W.shape}")
        print(f"   - Bias b: {b:.6f}")
        print(f"   - ||W||: {np.linalg.norm(W):.6f}")
        
        return W, b
    
    def predict_confidence(self, X):
        """
        Dự đoán confidence score từ W và b
        
        Args:
            X: Dữ liệu đầu vào
            
        Returns:
            confidence_scores: Điểm confidence
        """
        if self.W is None or self.b is None:
            raise ValueError("Hyperplane chưa được tính toán!")
        
        # Tính confidence: f(x) = W^T * x + b
        confidence_scores = np.dot(X, self.W) + self.b
        
        return confidence_scores
    
    def visualize_weight_vector(self, title="Weight Vector W"):
        """
        Trực quan hóa W như ảnh - sẽ thấy hình ảnh giống mặt người
        
        Args:
            title: Tiêu đề của plot
        """
        if self.W is None:
            raise ValueError("Hyperplane chưa được tính toán!")
        
        # Reshape W về kích thước ảnh
        W_img = self.W.reshape(self.patch_size)
        
        # Chuẩn hóa để hiển thị
        W_normalized = (W_img - W_img.min()) / (W_img.max() - W_img.min())
        
        plt.figure(figsize=(10, 8))
        
        # Plot W
        plt.subplot(2, 2, 1)
        plt.imshow(W_normalized, cmap='gray')
        plt.title(f'{title} (Normalized)')
        plt.colorbar()
        plt.axis('off')
        
        # Plot W với colormap khác
        plt.subplot(2, 2, 2)
        plt.imshow(W_img, cmap='RdBu_r', vmin=-np.abs(W_img).max(), vmax=np.abs(W_img).max())
        plt.title(f'{title} (Centered)')
        plt.colorbar()
        plt.axis('off')
        
        # Plot histogram của W
        plt.subplot(2, 2, 3)
        plt.hist(self.W, bins=50, alpha=0.7, color='blue')
        plt.title('Histogram of W values')
        plt.xlabel('Weight value')
        plt.ylabel('Frequency')
        
        # Plot magnitude của W theo vị trí
        plt.subplot(2, 2, 4)
        W_magnitude = np.abs(W_img)
        plt.imshow(W_magnitude, cmap='hot')
        plt.title('Magnitude of W')
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Weight vector đã được trực quan hóa!")
        print(f"   - Min W: {self.W.min():.6f}")
        print(f"   - Max W: {self.W.max():.6f}")
        print(f"   - Mean |W|: {np.mean(np.abs(self.W)):.6f}")
    
    def find_best_C(self, X_train, y_train, X_val, y_val, C_values=None):
        """
        Tìm giá trị C tốt nhất bằng cách thử các giá trị khác nhau
        
        Args:
            X_train, y_train: Dữ liệu huấn luyện
            X_val, y_val: Dữ liệu validation
            C_values: Danh sách giá trị C để thử
            
        Returns:
            best_C: Giá trị C tốt nhất
            best_accuracy: Accuracy tốt nhất
        """
        if C_values is None:
            C_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        
        print(f"Tìm giá trị C tốt nhất...")
        print(f"   - Các giá trị C sẽ thử: {C_values}")
        
        accuracies = []
        best_C = None
        best_accuracy = 0
        
        for C in C_values:
            print(f"\nThử C = {C}...")
            
            # Huấn luyện SVM với C hiện tại
            self.train_svm(X_train, y_train, C=C)
            
            # Đánh giá
            accuracy = self.evaluate_svm(X_val, y_val)
            accuracies.append(accuracy)
            
            # Cập nhật best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_C = C
                print(f"   C = {C} là tốt nhất cho đến nay!")
        
        # Huấn luyện lại với C tốt nhất
        print(f"\nHuấn luyện lại với C tốt nhất = {best_C}")
        self.train_svm(X_train, y_train, C=best_C)
        self.compute_hyperplane()
        
        # Vẽ biểu đồ accuracy vs C
        plt.figure(figsize=(10, 6))
        plt.plot(C_values, accuracies, 'bo-', linewidth=2, markersize=8)
        plt.axvline(x=best_C, color='red', linestyle='--', label=f'Best C = {best_C}')
        plt.xlabel('C (Regularization parameter)')
        plt.ylabel('Validation Accuracy')
        plt.title('Accuracy vs Regularization Parameter C')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.show()
        
        self.best_C = best_C
        self.best_accuracy = best_accuracy
        
        print(f"\nKết quả tìm C tốt nhất:")
        print(f"   - Best C: {best_C}")
        print(f"   - Best Accuracy: {best_accuracy:.4f}")
        
        return best_C, best_accuracy
    
    def analyze_regularization_effect(self, X_train, y_train, X_val, y_val):
        """
        Phân tích hiệu ứng của regularization
        
        Args:
            X_train, y_train: Dữ liệu huấn luyện
            X_val, y_val: Dữ liệu validation
        """
        print("Phân tích hiệu ứng regularization...")
        
        C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
        W_norms = []
        accuracies = []
        
        for C in C_values:
            print(f"\nPhân tích C = {C}...")
            
            # Huấn luyện SVM
            self.train_svm(X_train, y_train, C=C)
            
            # Tính W
            W, b = self.compute_hyperplane()
            
            # Tính ||W||
            W_norm = np.linalg.norm(W)
            W_norms.append(W_norm)
            
            # Tính accuracy
            accuracy = self.evaluate_svm(X_val, y_val)
            accuracies.append(accuracy)
            
            # Trực quan hóa W
            self.visualize_weight_vector(f"Weight Vector W (C={C})")
        
        # Vẽ biểu đồ phân tích
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy vs C
        axes[0].plot(C_values, accuracies, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('C (Regularization parameter)')
        axes[0].set_ylabel('Validation Accuracy')
        axes[0].set_title('Accuracy vs Regularization Parameter C')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')
        
        # ||W|| vs C
        axes[1].plot(C_values, W_norms, 'ro-', linewidth=2, markersize=8)
        axes[1].set_xlabel('C (Regularization parameter)')
        axes[1].set_ylabel('||W|| (Weight vector norm)')
        axes[1].set_title('Weight Vector Norm vs Regularization Parameter C')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nKết luận phân tích regularization:")
        print(f"   - C nhỏ → ||W|| nhỏ → Regularization mạnh → W giống mặt hơn")
        print(f"   - C lớn → ||W|| lớn → Regularization yếu → W có thể overfit")

if __name__ == "__main__":
    # Demo sử dụng
    from data_preparation import DataPreparation
    
    # Chuẩn bị dữ liệu
    data_prep = DataPreparation()
    X_train, X_val, y_train, y_val = data_prep.prepare_data()
    
    # Huấn luyện SVM
    svm_trainer = SVMTrainer()
    
    # Tìm C tốt nhất
    best_C, best_acc = svm_trainer.find_best_C(X_train, y_train, X_val, y_val)
    
    # Phân tích regularization
    svm_trainer.analyze_regularization_effect(X_train, y_train, X_val, y_val) 