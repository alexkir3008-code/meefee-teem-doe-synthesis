"""Та же физика и та же функция ошибки, что в numpy-версии, но на PyTorch (пакетно и с автоградиентом)."""
import numpy as np
import torch
import torch.nn.functional as Fn
from .config import N, DISTANCES
from .physics_np import CHIRPS
from .metrics_np import C1, C2

_cache = {}


def chirps(device):
    key = str(device)
    if key not in _cache:
        c = np.stack(CHIRPS)                                      # (4, N, N)
        _cache[key] = torch.complex(torch.tensor(c.real, dtype=torch.float32),
                                    torch.tensor(c.imag, dtype=torch.float32)).to(device)
    return _cache[key]


def intensity(phase):
    """phase (B, N, N) -> яркость во всех 4 плоскостях (B, 4, N, N), центрированная, как в numpy."""
    field = torch.complex(torch.cos(phase), torch.sin(phase))[:, None] * chirps(phase.device)[None]
    F = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1))), dim=(-2, -1))
    return F.real ** 2 + F.imag ** 2


def _gauss_kernel(device, sigma=1.5, radius=5):
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]                # (1,1,11,11)


def _G(a, k):
    b, c, h, w = a.shape
    a = Fn.pad(a.reshape(b * c, 1, h, w), (5, 5, 5, 5), mode="circular")
    return Fn.conv2d(a, k).reshape(b, c, h, w)


def to_display(I, T=None):
    """Как physics_np.to_display: модуль поля |K| = sqrt(I), нормировка на максимум. Результат 0..1."""
    A = torch.sqrt(I + 1e-12)
    return A / (A.amax(dim=(-2, -1), keepdim=True) + 1e-12)


def metrics(X01, T01):
    """КК и оконный SSIM для каждой пары: (B, 4) и (B, 4). Цель нормируется на максимум, шкала 0..255."""
    X = X01 * 255
    Y = T01 / (T01.amax(dim=(-2, -1), keepdim=True) + 1e-12) * 255
    xc = X - X.mean(dim=(-2, -1), keepdim=True)
    yc = Y - Y.mean(dim=(-2, -1), keepdim=True)
    r = (xc * yc).sum(dim=(-2, -1)) / torch.sqrt((xc ** 2).sum(dim=(-2, -1)) * (yc ** 2).sum(dim=(-2, -1)) + 1e-12)
    k = _gauss_kernel(X.device)
    mx, my = _G(X, k), _G(Y, k)
    sxx = _G(X * X, k) - mx ** 2
    syy = _G(Y * Y, k) - my ** 2
    sxy = _G(X * Y, k) - mx * my
    m = (2 * mx * my + C1) * (2 * sxy + C2) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return r, m.mean(dim=(-2, -1))


def _pcc(X, Y):
    xc = X - X.mean(dim=(-2, -1), keepdim=True)
    yc = Y - Y.mean(dim=(-2, -1), keepdim=True)
    return (xc * yc).sum(dim=(-2, -1)) / torch.sqrt((xc ** 2).sum(dim=(-2, -1)) * (yc ** 2).sum(dim=(-2, -1)) + 1e-12)


def _ssim(X, Y):
    k = _gauss_kernel(X.device)
    mx, my = _G(X, k), _G(Y, k)
    sxx = _G(X * X, k) - mx ** 2
    syy = _G(Y * Y, k) - my ** 2
    sxy = _G(X * Y, k) - mx * my
    m = (2 * mx * my + C1) * (2 * sxy + C2) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return m.mean(dim=(-2, -1))


def plane_losses(phase, T, w_ssim=0.5):
    """Ошибка по каждой плоскости (B, 4) — как optimize_np.plane_loss:
    (1 − КК(|K|, T)) + w·(1 − SSIM(|K| под среднюю яркость цели, T))."""
    A = torch.sqrt(intensity(phase) + 1e-12)
    Y = T / (T.amax(dim=(-2, -1), keepdim=True) + 1e-12) * 255
    L = 1 - _pcc(A, Y)
    if w_ssim > 0:
        s = Y.sum(dim=(-2, -1), keepdim=True) / 255 / (A.sum(dim=(-2, -1), keepdim=True) + 1e-12)
        X = torch.clamp(A * s, max=1.0) * 255
        L = L + w_ssim * (1 - _ssim(X, Y))
    return L


def loss_fn(phase, T, w_ssim=0.5, p=2.0):
    """Сумма по плоскостям L_k^p (p=2 сильнее тянет худшую картинку), среднее по пакету."""
    return (plane_losses(phase, T, w_ssim) ** p).sum(dim=1).mean()


def refine(T, init_phase, steps=300, lr=0.05, w_ssim=0.5, p=2.0):
    """Доводка маски градиентом по физике. T (B,4,N,N), init (B,N,N). Возвращает лучшую фазу для каждого примера."""
    phi = init_phase.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([phi], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=lr * 0.05)
    best = phi.detach().clone()
    best_l = torch.full((T.shape[0],), float("inf"), device=T.device)
    for _ in range(steps):
        L = (plane_losses(phi, T, w_ssim) ** p).sum(dim=1)        # по каждому примеру
        better = L.detach() < best_l
        best[better] = phi.detach()[better]
        best_l = torch.where(better, L.detach(), best_l)
        opt.zero_grad()
        L.sum().backward()
        opt.step()
        sched.step()
    return torch.remainder(best + np.pi, 2 * np.pi) - np.pi
