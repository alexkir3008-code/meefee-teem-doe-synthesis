"""ЗАПУСТИТЬ ПЕРВЫМ: проверяет, что всё работает на этом компьютере.
  python selftest.py
Если какая-то строка не OK — пришлите вывод целиком."""
import sys
import time
import numpy as np

print("Python", sys.version.split()[0])
from doe import physics_np as P
from doe.optimize_np import loss_grad
from doe.data import ImageSource
print("numpy-часть: OK")

try:
    import torch
except Exception as e:
    print("PyTorch НЕ установлен:", e); sys.exit(1)
print("PyTorch", torch.__version__)
from doe.device import pick_device
from doe import physics_torch as PT
from doe.model import DOENet, DOENetPhys, DOENetBP

dev = pick_device()
src = ImageSource(seed=3)
T = np.stack([src.quad() for _ in range(2)]).astype(np.float32)
rng = np.random.default_rng(0)
ph = rng.uniform(-np.pi, np.pi, (2, 128, 128)).astype(np.float32)

# 1) физика torch == numpy
It = PT.intensity(torch.tensor(ph, device=dev)).cpu().numpy()
for b in range(2):
    for k in range(4):
        In = P.intensity(ph[b].astype(np.float64), k)
        err = np.abs(It[b, k] / It[b, k].sum() - In / In.sum()).max() / (In / In.sum()).max()
        assert err < 1e-3, f"физика расходится: {err}"
print("физика PyTorch совпадает с numpy: OK")

# 2) функция ошибки torch == numpy
Lt = PT.loss_fn(torch.tensor(ph[:1], device=dev), torch.tensor(T[:1], device=dev)).item()
Ln, _ = loss_grad(ph[0].astype(np.float64), list(T[0].astype(np.float64)))
assert abs(Lt - Ln) < 2e-2 * max(1, abs(Ln)), (Lt, Ln)
print(f"функция ошибки совпадает (torch {Lt:.4f}, numpy {Ln:.4f}): OK")
Lt2 = PT.loss_fn(torch.tensor(ph[:1], device=dev), torch.tensor(T[:1], device=dev), w_max=0, w_q=0.5, w_bg=1.0).item()
Ln2, _ = loss_grad(ph[0].astype(np.float64), list(T[0].astype(np.float64)), w_max=0, w_q=0.5, w_bg=1.0)
assert abs(Lt2 - Ln2) < 2e-2 * max(1, abs(Ln2)), (Lt2, Ln2)
print(f"ошибка с перцентилем и фоном совпадает (torch {Lt2:.4f}, numpy {Ln2:.4f}): OK")

# 3) сеть и несколько шагов обучения
net = DOENet(32).to(dev)
opt = torch.optim.Adam(net.parameters(), 1e-3)
Tt = torch.tensor(T, device=dev)
l0 = None
t0 = time.time()
for i in range(30):
    loss = PT.loss_fn(net(Tt), Tt)
    opt.zero_grad(); loss.backward(); opt.step()
    l0 = l0 or loss.item()
print(f"обучение 30 шагов: ошибка {l0:.3f} -> {loss.item():.3f}, {time.time() - t0:.1f} с: OK")

# 4) сеть с физикой внутри: в начале совпадает с обычной сетью и учится
phys = DOENetPhys(32, stages=2).to(dev)
phys.base_net.load_state_dict(net.state_dict())
net.eval(); phys.eval()
with torch.no_grad():
    d = (torch.cos(phys(Tt)) - torch.cos(net(Tt))).abs().max().item()
assert d < 1e-3, f"сеть с физикой в начале должна совпадать с обычной: {d}"
phys.train()
opt2 = torch.optim.Adam(phys.parameters(), 1e-4)
t0 = time.time()
for i in range(10):
    loss2 = PT.loss_fn(phys(Tt), Tt)
    opt2.zero_grad(); loss2.backward(); opt2.step()
print(f"сеть с физикой внутри: 10 шагов {time.time() - t0:.1f} с: OK")

# 5) обратная дифракция: прямой расчёт от результата возвращает исходное поле
U = torch.complex(torch.rand(1, 4, 128, 128, device=dev), torch.rand(1, 4, 128, 128, device=dev))
Hk = PT.back_propagate(U)
for k in range(4):
    Fk = PT.field(Hk[:, k])[:, k]
    err = ((Fk - U[:, k]).abs().max() / U[:, k].abs().max()).item()
    assert err < 1e-3, f"обратная дифракция неточна: {err}"
print("обратная дифракция Френеля точна: OK")

# 6) сеть с обратной дифракцией учится
bp = DOENetBP(32, stages=0).to(dev)
opt3 = torch.optim.Adam(bp.parameters(), 1e-3)
l_first = None
t0 = time.time()
for i in range(40):
    loss3 = PT.loss_fn(bp(Tt), Tt)
    opt3.zero_grad(); loss3.backward(); opt3.step()
    l_first = l_first or loss3.item()
assert torch.isfinite(loss3), "ошибка не число"
print(f"сеть с обратной дифракцией: ошибка {l_first:.3f} -> {loss3.item():.3f}, {time.time() - t0:.1f} с: OK")

# 7) физические блоки: в начале «прозрачны», затем учатся
bp3 = DOENetBP(32, stages=3).to(dev)
bp3.load_state_dict(bp.state_dict(), strict=False)
bp.eval(); bp3.eval()
with torch.no_grad():
    d = (torch.cos(bp3(Tt)) - torch.cos(bp(Tt))).abs().max().item()
assert d < 1e-2, f"блоки в начале должны быть прозрачны: {d}"
bp3.train()
opt4 = torch.optim.Adam(bp3.parameters(), 3e-4)
l_first = None
t0 = time.time()
for i in range(20):
    loss4 = PT.loss_fn(bp3(Tt), Tt)
    opt4.zero_grad(); loss4.backward(); opt4.step()
    l_first = l_first or loss4.item()
assert torch.isfinite(loss4), "ошибка не число"
print(f"сеть с 3 физическими блоками: ошибка {l_first:.3f} -> {loss4.item():.3f}, {(time.time() - t0) / 20:.2f} с/шаг: OK")
print("\nВСЁ РАБОТАЕТ. Дальше: python train.py")
