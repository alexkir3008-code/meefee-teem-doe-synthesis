"""Метрики кейса (слайд 3): коэффициент корреляции Пирсона и индекс структурного сходства.

Считаются по 8-битным изображениям (0..255), как они сохраняются в png.
ИСС считается двумя способами:
  - issg  — ровно по формуле слайда (средние и дисперсии по всему изображению);
  - ssim  — стандартный оконный SSIM (Wang et al., 2004, гауссово окно 11×11, σ=1.5),
            как в skimage/MATLAB. В отчёт идут оба.
"""
import numpy as np

C1 = (0.01 * 255) ** 2
C2 = (0.03 * 255) ** 2


def as255(a01):
    return np.round(np.clip(a01, 0, 1) * 255)


def pearson(A, B):
    a = A - A.mean()
    b = B - B.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def issg(A, B):
    """ИСС по формуле слайда (глобальный)."""
    ma, mb = A.mean(), B.mean()
    va, vb = A.var(), B.var()
    cov = ((A - ma) * (B - mb)).mean()
    return float((2 * ma * mb + C1) * (2 * cov + C2) / ((ma**2 + mb**2 + C1) * (va + vb + C2)))


def _gauss_filter(img, sigma=1.5, radius=5):
    x = np.arange(-radius, radius + 1)
    g = np.exp(-x**2 / (2 * sigma**2)); g /= g.sum()
    pad = np.pad(img, radius, mode="reflect")
    tmp = np.apply_along_axis(lambda r: np.convolve(r, g, mode="valid"), 1, pad)
    return np.apply_along_axis(lambda c: np.convolve(c, g, mode="valid"), 0, tmp)


def ssim(A, B):
    """Оконный SSIM (гауссово окно), диапазон данных 255."""
    A = A.astype(np.float64); B = B.astype(np.float64)
    mu_a, mu_b = _gauss_filter(A), _gauss_filter(B)
    saa = _gauss_filter(A * A) - mu_a**2
    sbb = _gauss_filter(B * B) - mu_b**2
    sab = _gauss_filter(A * B) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a**2 + mu_b**2 + C1) * (saa + sbb + C2))
    return float(m.mean())


def evaluate(targets01, recons01):
    """Попарные метрики для 4 пар (исходное, восстановленное).
    Оба изображения нормируются на максимум (= 1), как сказал эксперт, затем переводятся в 0..255."""
    rows = []
    for k, (t, r) in enumerate(zip(targets01, recons01)):
        A = as255(t / (t.max() + 1e-12))
        B = as255(r / (r.max() + 1e-12))
        rows.append({"img": k + 1, "KK": pearson(A, B), "ISS_slide": issg(A, B), "SSIM": ssim(A, B)})
    return rows
