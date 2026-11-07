"""Прямая градиентная оптимизация маски на numpy — ровно под метрики кейса (КК + ИСС).

Нужна для проверки подхода и как запасной вариант без PyTorch.
Градиенты выведены аналитически и проверены численно (test_np.py).
"""
import numpy as np
from .physics_np import CHIRPS
from .metrics_np import C1, C2

_M = 128
_x = np.fft.fftfreq(_M) * _M
_g = np.exp(-_x**2 / (2 * 1.5**2)); _g /= _g.sum()
_GK = np.fft.fft2(np.outer(_g, _g)).real          # гауссово окно 1.5 (круговая свёртка, симметрична)


def _G(a):
    return np.real(np.fft.ifft2(np.fft.fft2(a) * _GK))


def _ssim_grad(x, y):
    """SSIM(x, y) и dSSIM/dx (оконный, гауссово окно σ=1.5)."""
    mx, my = _G(x), _G(y)
    sxx = _G(x * x) - mx**2; syy = _G(y * y) - my**2; sxy = _G(x * y) - mx * my
    A1 = 2 * mx * my + C1; A2 = 2 * sxy + C2
    B1 = mx**2 + my**2 + C1; B2 = sxx + syy + C2
    m = A1 * A2 / (B1 * B2)
    a = m * (2 * my / A1 - 2 * my / A2 - 2 * mx / B1 + 2 * mx / B2)
    b = 2 * m / A2
    c = -2 * m / B2
    grad = (_G(a) + y * _G(b) + x * _G(c)) / m.size
    return m.mean(), grad


def _pcc_grad(x, y):
    xc = x - x.mean(); yc = y - y.mean()
    nx = np.sqrt((xc**2).sum()) + 1e-12; ny = np.sqrt((yc**2).sum()) + 1e-12
    r = (xc * yc).sum() / (nx * ny)
    return r, yc / (nx * ny) - r * xc / nx**2


Q_ROBUST = 0.995                                               # «почти максимум»: 99,5-й перцентиль


def plane_loss(A, T, w_ssim=0.5, w_max=0.5, w_q=0.0):
    """A — модуль поля |K| (не сдвинут), T — цель (не сдвинута).
    Ошибка = (1 − КК(A, T)) + w·(1 − SSIM(A', T)), где A' — модуль, приведённый к средней яркости цели
    и обрезанный сверху. КК не зависит от масштаба. SSIM на нормировке по средней яркости нельзя
    «обмануть» одной яркой точкой (при нормировке на максимум простая линза даёт SSIM 0,84 без изображения).
    Возвращает ошибку и dОшибка/dA."""
    Y = T / (T.max() + 1e-12) * 255
    r, gr = _pcc_grad(A, Y)
    L = 1 - r
    gA = -gr
    q = 0.0
    if w_ssim > 0:
        S = A.sum() + 1e-12
        s = Y.sum() / 255 / S
        X = s * A
        free = X < 1.0
        q, gq = _ssim_grad(np.minimum(X, 1.0) * 255, Y)
        L += w_ssim * (1 - q)
        gX = -w_ssim * gq * 255 * free
        gA = gA + s * gX - (gX * X).sum() / S
    if w_max > 0:                                              # SSIM как у проверяющих (нормировка на максимум)
        j = np.unravel_index(np.argmax(A), A.shape)
        M = A[j] + 1e-12
        qm, gq = _ssim_grad(A / M * 255, Y)
        L += w_max * (1 - qm)
        gX = -w_max * gq * 255
        g2 = gX / M
        g2[j] -= (gX * A).sum() / M**2
        gA = gA + g2
    if w_q > 0:                                                # SSIM при нормировке на 99,5-й перцентиль:
        flat = A.ravel()                                       # одна яркая точка её не сдвигает
        k = int(Q_ROBUST * (flat.size - 1))
        idx = np.argpartition(flat, k)[k]
        M = flat[idx] + 1e-12
        Xr = A / M
        free = Xr < 1.0
        qq, gq = _ssim_grad(np.minimum(Xr, 1.0) * 255, Y)
        L += w_q * (1 - qq)
        gX = -w_q * gq * 255 * free
        g2 = gX / M
        g2.ravel()[idx] -= (gX * A).sum() / M**2
        gA = gA + g2
    return L, gA, r, q


def loss_grad(phase, targets, w_ssim=0.5, p=2.0, w_max=0.5, w_q=0.0, w_bg=0.0):
    """Сумма по плоскостям L_k^p (p=2 сильнее тянет худшую картинку) и градиент по фазе."""
    Ls, grads = [], []
    for k, T in enumerate(targets):
        u = np.fft.ifftshift(np.exp(1j * phase) * CHIRPS[k])
        F = np.fft.fft2(u)
        I = np.abs(F) ** 2
        A = np.sqrt(I + 1e-12)                                   # модуль поля
        Tu = np.fft.ifftshift(T)
        L, gA, _, _ = plane_loss(A, Tu, w_ssim, w_max, w_q)
        gI = gA / (2 * A)                                        # dL/dI = dL/dA · dA/dI
        if w_bg > 0:                                             # доля света вне изображения (фон)
            m = (Tu < 0.05 * Tu.max()).astype(float)
            S = I.sum() + 1e-12
            f = (m * I).sum() / S
            L += w_bg * f
            gI = gI + w_bg * (m - f) / S
        g = -2 * np.imag(u * np.fft.fft2(gI * np.conj(F)))
        Ls.append(L); grads.append(np.fft.fftshift(g))
    Ls = np.array(Ls)
    w = p * Ls ** (p - 1)
    return float((Ls**p).sum()), sum(wi * gi for wi, gi in zip(w, grads))


def optimize(targets, init=None, steps=400, lr=0.05, w_ssim=0.5, p=2.0, seed=0):
    rng = np.random.default_rng(seed)
    phase = init.copy() if init is not None else rng.uniform(-np.pi, np.pi, targets[0].shape)
    m = np.zeros_like(phase); v = np.zeros_like(phase)
    for t in range(1, steps + 1):
        _, g = loss_grad(phase, targets, w_ssim, p)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        phase = phase - lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return np.angle(np.exp(1j * phase))
