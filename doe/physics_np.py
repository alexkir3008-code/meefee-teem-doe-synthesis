"""Физика на numpy: прямой расчёт дифракции Френеля с одним преобразованием Фурье.

Формула со слайда 2:
  K(x,y) = exp(jkz)/(jλz) · exp(jπ(x²+y²)/(λz)) · FFT( H(u,v) · exp(jπ(u²+v²)/(λz)) )
H = exp(jφ) — поле после фазового ДОЭ. Восстановленное изображение = |K| (модуль, по условию кейса).
"""
import numpy as np
from .config import N, PIXEL, WAVELENGTH, DISTANCES


def coords(n=N, d=PIXEL):
    c = (np.arange(n) - n // 2) * d
    return np.meshgrid(c, c)


def chirp(z, n=N, d=PIXEL, wl=WAVELENGTH):
    """exp(jπ(u²+v²)/(λz)) в плоскости ДОЭ."""
    U, V = coords(n, d)
    return np.exp(1j * np.pi * (U**2 + V**2) / (wl * z))


CHIRPS = [chirp(z) for z in DISTANCES]


def out_pixel(z, n=N, d=PIXEL, wl=WAVELENGTH):
    """Шаг сетки в плоскости восстановления: λz/(N·d)."""
    return wl * z / (n * d)


def fresnel_field(phase, k):
    """Полное комплексное поле K(x,y) в плоскости k по формуле слайда."""
    z = DISTANCES[k]
    kw = 2 * np.pi / WAVELENGTH
    dxo = out_pixel(z)
    X, Y = coords(N, dxo)
    pre = np.exp(1j * kw * z) / (1j * WAVELENGTH * z) * np.exp(1j * np.pi * (X**2 + Y**2) / (WAVELENGTH * z))
    spec = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.exp(1j * phase) * CHIRPS[k])))
    return pre * spec


def intensity(phase, k):
    """|K|² без постоянных множителей (на форму изображения они не влияют)."""
    spec = np.fft.fft2(np.fft.ifftshift(np.exp(1j * phase) * CHIRPS[k]))
    return np.abs(np.fft.fftshift(spec)) ** 2


def amplitude(phase, k):
    """Модуль поля |K| (по условию кейса сохраняются модули значений)."""
    return np.sqrt(intensity(phase, k))


def reconstruct_all(phase):
    """Модули поля |K| во всех 4 плоскостях."""
    return [amplitude(phase, k) for k in range(len(DISTANCES))]


def norm_max(a):
    """Нормировка на максимум (ответ эксперта: каждое изображение перед сравнением нормируется на 1)."""
    return a / (a.max() + 1e-12)


def to_display(A, target=None):
    """Модуль поля -> 0..1, нормировка на максимум."""
    return norm_max(A)


# ---------- классические методы (базовая линия) ----------

def _forward(phase_field, k):
    return np.fft.fft2(np.fft.ifftshift(phase_field * CHIRPS[k]))


def _backward(spec, k):
    return np.fft.fftshift(np.fft.ifft2(spec)) * np.conj(CHIRPS[k])


def gs_multiplane(targets, iterations=300, seed=0, weighted=True):
    """Гершберг–Сакстон для 4 плоскостей (глобальный, с усреднением полей).
    weighted=True: веса плоскостей подстраиваются, чтобы все 4 изображения вышли одинаково хорошими."""
    rng = np.random.default_rng(seed)
    amps = [np.fft.ifftshift(t / (t.max() + 1e-12)) for t in targets]   # нужный модуль = яркость картинки
    w = np.ones(len(targets))
    phase = rng.uniform(-np.pi, np.pi, (N, N))
    for it in range(iterations):
        acc = np.zeros((N, N), complex)
        quality = np.zeros(len(targets))
        for k, a in enumerate(amps):
            F = _forward(np.exp(1j * phase), k)
            mag = np.abs(F)
            quality[k] = np.corrcoef(mag.ravel(), a.ravel())[0, 1]
            back = _backward(a * np.exp(1j * np.angle(F)), k)
            acc += w[k] * back / (np.abs(back).mean() + 1e-12)
        phase = np.angle(acc)
        if weighted and it % 10 == 9:
            w *= np.exp(2.0 * (quality.mean() - quality))
            w /= w.mean()
    return phase


# ---------- формат файлов кейса ----------

def phase_to_uint8(phase):
    """0 соответствует −π, 255 соответствует +π."""
    p = np.angle(np.exp(1j * phase))             # в диапазон [−π, π]
    return np.round((p + np.pi) / (2 * np.pi) * 255).astype(np.uint8)


def uint8_to_phase(img):
    return img.astype(np.float64) / 255 * 2 * np.pi - np.pi


def load_gray(path):
    from PIL import Image
    im = Image.open(path).convert("L")
    if im.size != (N, N):
        im = im.resize((N, N), Image.BICUBIC)
    return np.asarray(im, dtype=np.float64) / 255


def save_gray(arr01, path):
    from PIL import Image
    Image.fromarray(np.round(np.clip(arr01, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


def save_doe(phase, path):
    from PIL import Image
    Image.fromarray(phase_to_uint8(phase), mode="L").save(path)
