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


class UNetCore(nn.Module):
    """Та же U-Net, но с произвольным числом входов и выходов (для блоков уточнения)."""

    def __init__(self, cin, cout, base=16):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8, base * 8]
        self.inc = Block(cin, ch[0])
        self.down = nn.ModuleList([Block(ch[i], ch[i + 1]) for i in range(4)])
        self.mid = Block(ch[4], ch[4])
        self.up = nn.ModuleList([Block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(4))])
        self.out = nn.Conv2d(ch[0], cout, 1)
        nn.init.zeros_(self.out.weight)          # в начале блок ничего не меняет — нет ухудшения
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.down:
            skips.append(d(Fn.avg_pool2d(skips[-1], 2)))
        h = self.mid(skips.pop())
        for u in self.up:
            s = skips.pop()
            h = u(torch.cat([Fn.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False), s], dim=1))
        return self.out(h)


class DOENetPhys(nn.Module):
    """Нейросеть с физической моделью Френеля внутри.

    1) DOENet выдаёт первую маску.
    2) Физический слой (дифракция Френеля, без обучаемых параметров) считает,
       что эта маска даёт в 4 плоскостях.
    3) Блок-сеть видит цели, полученные изображения и маску и выдаёт поправку к маске.
    Шаги 2–3 повторяются фиксированное число раз (stages) с РАЗНЫМИ обученными блоками.
    Это один проход сети с фиксированной архитектурой: веса обучены заранее,
    под конкретные входные картинки ничего не подбирается.
    """

    def __init__(self, base=32, stages=2, refiner_base=16):
        super().__init__()
        self.base_net = DOENet(base)
        self.refiners = nn.ModuleList([UNetCore(4 + 4 + 2 + 3, 1, refiner_base) for _ in range(stages)])

    def forward(self, T):
        from .physics_torch import intensity
        phi = self.base_net(T)
        coords = self.base_net.coords.expand(T.shape[0], -1, -1, -1)
        for r in self.refiners:
            A = torch.sqrt(intensity(phi) + 1e-12)
            A = A / (A.amax(dim=(-2, -1), keepdim=True) + 1e-12)
            x = torch.cat([T * 2 - 1, A * 2 - 1, torch.cos(phi)[:, None], torch.sin(phi)[:, None], coords], dim=1)
            phi = phi + r(x)[:, 0]
        return torch.atan2(torch.sin(phi), torch.cos(phi))


def build_model(base=32, stages=0, refiner_base=16):
    return DOENet(base) if stages == 0 else DOENetPhys(base, stages, refiner_base)


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    net = build_model(ckpt.get("base", 32), ckpt.get("stages", 0), ckpt.get("refiner_base", 16)).to(device)
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
