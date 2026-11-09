"""HOG против сырых пикселей: сверка со scikit-image и устойчивость к сдвигу картинки."""
import numpy as np
from PIL import Image
from scipy.ndimage import shift as nd_shift
from skimage.feature import hog as sk_hog
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hog import color_histogram, hog

digits = load_digits()
imgs = np.stack([np.asarray(Image.fromarray((im * 16).astype(np.uint8)).resize((32, 32), Image.Resampling.BILINEAR), float)
                 for im in digits.images])                                     # 8x8 -> 32x32, иначе HOG-клеткам негде жить
y = digits.target

# 1. Сверка со scikit-image на одной картинке
ours = hog(imgs[0], orientations=9, cell=8, block=2)
ref = sk_hog(imgs[0], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
print(f"длина HOG: своя {ours.size}, skimage {ref.size} | корреляция {np.corrcoef(ours, ref)[0, 1]:.4f} | max|разница| {np.abs(ours - ref).max():.3f}")

# 2. Обучаем на исходных картинках, тестируем на сдвинутых на 2 пикселя (камера чуть повернулась)
Xtr_img, Xte_img, ytr, yte = train_test_split(imgs, y, test_size=0.3, random_state=0, stratify=y)
shifted = np.stack([nd_shift(im, (2, -2), order=0) for im in Xte_img])
feats = {"пиксели": lambda ims: ims.reshape(len(ims), -1),
         "HOG": lambda ims: np.stack([hog(im) for im in ims])}
for name, f in feats.items():
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(f(Xtr_img), ytr)
    print(f"{name:<8} размерность {f(Xtr_img[:1]).shape[1]:>4} | test {model.score(f(Xte_img), yte):.3f} | test со сдвигом 2px {model.score(f(shifted), yte):.3f}")

# 3. Цветовая гистограмма: «отпечаток» сцены, не зависящий от расположения объектов
from sklearn.datasets import load_sample_images
china, flower = load_sample_images().images
h_c, h_f, h_c_flip = color_histogram(china), color_histogram(flower), color_histogram(china[:, ::-1])
print(f"\nцветовые гистограммы: china vs отражённый china — {np.abs(h_c - h_c_flip).sum():.3f}, china vs flower — {np.abs(h_c - h_f).sum():.3f} (L1, макс. 2)")
