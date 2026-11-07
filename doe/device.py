"""Выбор устройства: работает на любой видеокарте и без неё.

Порядок: NVIDIA/AMD (cuda; AMD — через сборку PyTorch с ROCm) → Intel (xpu) → Apple (mps) → процессор.
Перед использованием видеокарта проверяется на реальном расчёте FFT; если что-то не так — берётся процессор.
Принудительно: переменная окружения DOE_DEVICE=cpu (или cuda, mps, xpu).
"""
import os
import torch


def _works(dev):
    """Пробный расчёт той же физики и ошибки, что в решении, с обратным проходом."""
    try:
        from . import physics_torch as PT
        ph = (torch.rand(1, 128, 128, device=dev) * 6.28 - 3.14).requires_grad_(True)
        T = torch.rand(1, 4, 128, 128, device=dev)
        loss = PT.loss_fn(ph, T)
        loss.backward()
        return bool(torch.isfinite(loss)) and bool(torch.isfinite(ph.grad).all())
    except Exception as e:
        print(f"  {dev}: не подходит ({type(e).__name__}: {str(e)[:80]})")
        return False


def pick_device(verbose=True):
    forced = os.environ.get("DOE_DEVICE")
    candidates = [forced] if forced else []
    if torch.cuda.is_available():
        candidates.append("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        candidates.append("xpu")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        candidates.append("mps")
    candidates.append("cpu")
    for c in candidates:
        if c and _works(c):
            dev = torch.device(c)
            if verbose:
                name = torch.cuda.get_device_name(0) if c == "cuda" else c
                print(f"Устройство: {c} ({name})")
            if c == "cuda":
                torch.backends.cudnn.benchmark = True
            if c == "cpu":
                torch.set_num_threads(max(1, os.cpu_count() or 1))
            return dev
    return torch.device("cpu")
