"""Нейросеть: 4 изображения -> фазовая маска ДОЭ 128×128.

Архитектура U-Net (как в DeepCGH, Eybposh et al., Optics Express 2020 — 3D-голография по нескольким плоскостям).
Вход: 4 канала-изображения + 3 канала координат (x, y, x²+y²) — сети нужно знать положение пикселя,
потому что квадратичная фаза Френеля зависит от координаты.
Выход: 2 канала (cos, sin) -> фаза через atan2, без разрыва на ±π.
Обучение без готовых ответов: ошибку считает физика (дифракция Френеля) + метрики кейса.
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from .config import N


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
        )
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class DOENet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8, base * 8]   # 128→64→32→16→8
        self.inc = Block(7, ch[0])
        self.down = nn.ModuleList([Block(ch[i], ch[i + 1]) for i in range(4)])
        self.mid = Block(ch[4], ch[4])
        self.up = nn.ModuleList([Block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(4))])
        self.out = nn.Conv2d(ch[0], 2, 1)
        c = torch.linspace(-1, 1, N)
        yy, xx = torch.meshgrid(c, c, indexing="ij")
        self.register_buffer("coords", torch.stack([xx, yy, xx ** 2 + yy ** 2])[None])

    def forward(self, T):
        """T (B, 4, N, N) в 0..1 -> фаза (B, N, N) в (−π, π]."""
        x = torch.cat([T * 2 - 1, self.coords.expand(T.shape[0], -1, -1, -1)], dim=1)
        skips = [self.inc(x)]
        for d in self.down:
            skips.append(d(Fn.avg_pool2d(skips[-1], 2)))
        h = self.mid(skips.pop())
        for u in self.up:
            s = skips.pop()
            h = u(torch.cat([Fn.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False), s], dim=1))
        o = self.out(h)
        return torch.atan2(o[:, 1], o[:, 0])


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    net = DOENet(ckpt.get("base", 32)).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net


def default_model_path():
    """Своя свежая модель (checkpoints/best.pt), если есть; иначе — модель из репозитория (models/best.pt)."""
    import os
    for p in ("checkpoints/best.pt", "models/best.pt"):
        if os.path.exists(p):
            return p
    return "checkpoints/best.pt"
