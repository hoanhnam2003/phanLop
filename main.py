import os
import numpy as np
from skimage.io import imread
from skimage.feature import hog
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from skimage.transform import resize

def load_and_preprocess_dental_images(folder_path):
    hog_features = []
    labels = []
    target_size = (128, 128)  # Đặt kích thước cố định cho ảnh (tùy chọn)

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = imread(image_path)

            # Kiểm tra nếu ảnh có nhiều kênh, lấy kênh đầu tiên
            if image.ndim == 3:
                image = image[:, :, 0]

            # Resize ảnh
            image_resized = resize(image, target_size, anti_aliasing=True)

            # Tính toán đặc trưng HOG cho ảnh đã resize
            feature = hog(image_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, channel_axis=None)
            hog_features.append(feature)

            # Thêm nhãn ngẫu nhiên
            labels.append(np.random.randint(0, 3))

    return np.array(hog_features), np.array(labels)

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted')
    }

def main():
    # Tải và chuẩn bị bộ dữ liệu Iris
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

    # Tải và xử lý bộ ảnh nha khoa
    image_folder_path = r"C:\Users\Admin\Downloads\anhtest"  # Thay thế bằng đường dẫn tới thư mục ảnh của bạn
    X_dental, y_dental = load_and_preprocess_dental_images(image_folder_path)
    X_train_dental, X_test_dental, y_train_dental, y_test_dental = train_test_split(X_dental, y_dental, test_size=0.3, random_state=42)

    # Tạo các mô hình
    models = {
        "Naive Bayes": GaussianNB(),
        "CART (Gini Index)": DecisionTreeClassifier(criterion="gini"),
        "ID3 (Information Gain)": DecisionTreeClassifier(criterion="entropy"),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    }

    # Huấn luyện và đánh giá mô hình trên cả hai bộ dữ liệu
    for dataset_name, (X_train, X_test, y_train, y_test) in {
        "Iris Dataset": (X_train_iris, X_test_iris, y_train_iris, y_test_iris),
        "Dental Image Dataset": (X_train_dental, X_test_dental, y_train_dental, y_test_dental)
    }.items():
        print(f"\n--- Results for {dataset_name} ---")
        for model_name, model in models.items():
            metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
            print(f"\nModel: {model_name}")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
