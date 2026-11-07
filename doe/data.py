"""Обучающие изображения 128×128, 8 бит: цифры и буквы, фигуры, точки, полутоновые пятна.
Генерируются на лету, скачивать ничего не нужно. Можно подмешать свою папку с картинками."""
import os
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

N = 128
CHARS = "0123456789ABCDEFGHKLMNPRSTUVWXYZ"
_fonts = {}


def _font(size):
    if size not in _fonts:
        try:
            _fonts[size] = ImageFont.load_default(size=size)   # Pillow >= 10.1, работает везде
        except TypeError:
            _fonts[size] = ImageFont.load_default()
    return _fonts[size]


def _text(rng):
    im = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(im)
    count = int(rng.integers(1, 4))
    size = int(rng.integers(50, 110)) if count == 1 else int(rng.integers(34, 60))
    s = "".join(rng.choice(list(CHARS), count))
    font = _font(size)
    box = d.textbbox((0, 0), s, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    x = int(rng.integers(0, max(1, N - w))) - box[0]
    y = int(rng.integers(0, max(1, N - h))) - box[1]
    d.text((x, y), s, fill=int(rng.integers(160, 256)), font=font)
    if rng.random() < 0.5:
        im = im.rotate(float(rng.uniform(-30, 30)), resample=Image.BILINEAR)
    return im


def _shapes(rng):
    im = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(im)
    for _ in range(int(rng.integers(1, 5))):
        c = int(rng.integers(120, 256))
        x0, y0 = rng.integers(8, N - 40, 2)
        w, h = rng.integers(12, 56, 2)
        box = [int(x0), int(y0), int(min(N - 4, x0 + w)), int(min(N - 4, y0 + h))]
        kind = int(rng.integers(0, 5))
        width = int(rng.integers(2, 7))
        fill = c if rng.random() < 0.5 else None
        if kind == 0:
            d.ellipse(box, fill=fill, outline=c, width=width)
        elif kind == 1:
            d.rectangle(box, fill=fill, outline=c, width=width)
        elif kind == 2:
            pts = [tuple(int(v) for v in rng.integers(8, N - 8, 2)) for _ in range(3)]
            d.polygon(pts, fill=fill, outline=c)
        elif kind == 3:
            d.line([tuple(int(v) for v in rng.integers(8, N - 8, 2)) for _ in range(2)], fill=c, width=width)
        else:
            cx, cy, r = int(box[0] + 20), int(box[1] + 20), int(rng.integers(8, 20))
            pts = []
            for i in range(10):
                a = np.pi / 5 * i
                rr = r if i % 2 == 0 else r / 2.3
                pts.append((cx + rr * np.sin(a), cy - rr * np.cos(a)))
            d.polygon(pts, fill=c)
    return im


def _dots(rng):
    a = np.zeros((N, N))
    step = int(rng.integers(8, 20))
    off = int(rng.integers(16, 32))
    a[off:N - off:step, off:N - off:step] = rng.uniform(0.5, 1.0)
    im = Image.fromarray((a * 255).astype(np.uint8))
    if rng.random() < 0.5:
        im = im.filter(ImageFilter.MaxFilter(3))
    return im


def _blobs(rng):
    yy, xx = np.mgrid[0:N, 0:N]
    a = np.zeros((N, N))
    for _ in range(int(rng.integers(2, 7))):
        cx, cy = rng.uniform(20, N - 20, 2)
        s = rng.uniform(4, 18)
        a += rng.uniform(0.3, 1.0) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    a /= a.max()
    return Image.fromarray((a * 255).astype(np.uint8))


class ImageSource:
    """Случайные картинки-цели. folder — необязательная папка своих картинок (png/jpg)."""

    def __init__(self, seed=0, folder=None, folder_prob=0.3):
        self.rng = np.random.default_rng(seed)
        self.files = []
        if folder and os.path.isdir(folder):
            for ext in ("png", "jpg", "jpeg", "bmp"):
                self.files += glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True)
        self.folder_prob = folder_prob if self.files else 0.0

    def one(self):
        rng = self.rng
        if self.files and rng.random() < self.folder_prob:
            im = Image.open(self.files[int(rng.integers(len(self.files)))]).convert("L").resize((N, N), Image.BICUBIC)
        else:
            kind = rng.choice(["text", "shapes", "dots", "blobs"], p=[0.4, 0.35, 0.1, 0.15])
            im = {"text": _text, "shapes": _shapes, "dots": _dots, "blobs": _blobs}[kind](rng)
        if rng.random() < 0.5:
            im = im.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 1.2))))
        a = np.asarray(im, dtype=np.float32) / 255
        if a.max() < 0.05:                       # пустая картинка — заменить
            return self.one()
        return np.round(a * 255) / 255           # как 8-битный png

    def quad(self):
        """4 изображения — один пример для сети."""
        return np.stack([self.one() for _ in range(4)])

    def batch(self, b):
        return np.stack([self.quad() for _ in range(b)])
