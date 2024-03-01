import os
import cv2
import numpy as np

train_dosyasi = r"C:\Users\begum\Desktop\Projects\SIFT\data\data\train"
test_dosyasi = r"C:\Users\begum\Desktop\Projects\SIFT\data\data\test"

sift_train_dosyasi = r"C:\Users\begum\Desktop\Projects\SIFT\SIFT\train"
sift_test_dosyasi = r"C:\Users\begum\Desktop\Projects\SIFT\SIFT\test"


def extract_sift_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # cv2.imread fonksiyonu, belirtilen dosya yolu üzerindeki bir görüntüyü okur.
    # cv2.IMREAD_GRAYSCALE, görüntüyü gri tonlamalı olarak okumasını sağlar.
    sift = cv2.SIFT_create()  # SIFT algoritmasını uygulamak için OpenCV'nin SIFT sınıfını oluşturur.
    keypoints, descriptors = sift.detectAndCompute(img,
                                                   None)  # SIFT algoritmasını kullanarak görüntüdeki keypoints'leri ve bunları tanımlayan descriptors'ları bulur.
    return descriptors


# Bu fonksiyon, bir görüntü veri setinden SIFT özelliklerini çıkarır ve bu özellikleri Numpy formatında kaydederek bir çıkış klasörüne yazdırır.
def save_sift_features(image_folder, output_folder):
    for class_folder in os.listdir(
            image_folder):  # Görüntülerin bulunduğu ana klasördeki sınıfların (class) listesini alır ve her bir sınıf üzerinde döngü kurar
        if class_folder == ".DS_Store":
            continue  # macOS'ta bulunan gereksiz .DS_Store dosyasını atlamak için kontrol.
        class_path = os.path.join(image_folder,
                                  class_folder)  # Sınıfa ait görüntülerin bulunduğu klasörün yolunu oluşturur.
        output_class_folder = os.path.join(output_folder,
                                           class_folder)  # Çıkış klasöründeki sınıf klasörünün yolunu oluşturur. Bu klasör, sınıfa ait SIFT özelliklerinin kaydedileceği yerdir.
        os.makedirs(output_class_folder,
                    exist_ok=True)  #: Çıkış klasörünü oluşturur. exist_ok=True parametresi, klasörün zaten var olup olmadığını kontrol eder ve varsa tekrar oluşturmaz.

        for image_name in os.listdir(
                class_path):  # Her bir sınıfa ait görüntülerin listesini alır ve her bir görüntü üzerinde döngü kurar
            if image_name == ".DS_Store":
                continue  # .DS_Store dosyasını atla
            image_path = os.path.join(class_path, image_name)  # Görüntünün tam dosya yolunu oluşturur.
            features = extract_sift_features(
                image_path)  # 'extract_sift_features fonksiyonunu kullanarak görüntüden SIFT özelliklerini çıkarır.
            output_path = os.path.join(output_class_folder, f"{os.path.splitext(image_name)[0]}.npy")

            with open(output_path, 'wb') as f:
                np.save(f, features)


save_sift_features(train_dosyasi, sift_train_dosyasi)
save_sift_features(test_dosyasi, sift_test_dosyasi)
