"""Обучение нейросети синтеза ДОЭ.

  python train.py                      # обучение по умолчанию (~20 000 шагов)
  python train.py --steps 60000        # дольше = лучше (на ночь)
  python train.py --resume             # продолжить с последней точки
  python train.py --images my_pics     # подмешать свои картинки (папка png/jpg)
  python train.py --init checkpoints_old/best.pt --steps 15000 --lr 2e-4   # дообучить готовую модель
  python train.py --init models/best.pt --stages 2 --steps 15000 --lr 2e-4 # сеть с физикой внутри

Сохраняет checkpoints/best.pt (лучшая по проверке), checkpoints/last.pt и train_log.csv.
Работает на любой видеокарте или процессоре (см. doe/device.py).
"""
import argparse
import csv
import math
import os
import time
import numpy as np
import torch
from doe.device import pick_device
from doe.data import ImageSource
from doe.model import DOENet, build_model
from doe import physics_torch as PT
from doe import physics_np as P
from doe.metrics_np import evaluate


def make_cache(n, seed, folder):
    path = f"data_cache_{n}_{seed}.npy"
    if os.path.exists(path) and folder is None:
        return np.load(path)
    print(f"Генерирую {n} обучающих картинок (один раз)...")
    src = ImageSource(seed=seed, folder=folder)
    t0 = time.time()
    arr = np.stack([np.round(src.one() * 255).astype(np.uint8) for _ in range(n)])
    print(f"  готово за {time.time() - t0:.0f} с")
    if folder is None:
        np.save(path, arr)
    return arr


def sample(cache, b, rng, device):
    idx = rng.integers(0, len(cache), size=(b, 4))
    x = cache[idx].astype(np.float32) / 255                       # (b, 4, N, N)
    for i in range(b):                                             # простая аугментация
        for j in range(4):
            k = int(rng.integers(4))
            if k:
                x[i, j] = np.rot90(x[i, j], k).copy()
            if rng.random() < 0.5:
                x[i, j] = x[i, j, :, ::-1].copy()
    return torch.from_numpy(np.ascontiguousarray(x)).to(device)


def validate(net, val, device):
    net.eval()
    with torch.no_grad():
        ph = net(torch.from_numpy(val).to(device)).cpu().numpy()
    kk, ss = [], []
    for T, p in zip(val, ph):
        p = P.uint8_to_phase(P.phase_to_uint8(p))                  # как после сохранения в png
        R = [P.to_display(I, t) for I, t in zip(P.reconstruct_all(p), T)]
        rows = evaluate(list(T), R)
        kk += [r["KK"] for r in rows]
        ss += [r["SSIM"] for r in rows]
    net.train()
    return float(np.mean(kk)), float(np.mean(ss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--base", type=int, default=32, help="ширина сети (32 — норм, 48 — мощнее и медленнее)")
    ap.add_argument("--w-ssim", type=float, default=0.5, help="вес ИСС в ошибке (меньше — выше КК)")
    ap.add_argument("--cache", type=int, default=20000, help="сколько картинок сгенерировать для обучения")
    ap.add_argument("--images", default=None, help="папка своих картинок для подмешивания")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default=None, help="начать с весов готовой модели (дообучение), например checkpoints_old/best.pt")
    ap.add_argument("--stages", type=int, default=0, help="сколько физических блоков уточнения внутри сети (0 — обычная сеть, 2 — с физикой внутри)")
    ap.add_argument("--refiner-base", type=int, default=16, help="ширина блоков уточнения")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = pick_device()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    cache = make_cache(a.cache, a.seed, a.images)
    vs = ImageSource(seed=12345)
    val = np.stack([vs.quad() for _ in range(16)]).astype(np.float32)

    os.makedirs("checkpoints", exist_ok=True)
    if a.init and not a.resume:
        a.base = torch.load(a.init, map_location="cpu").get("base", a.base)
    net = build_model(a.base, a.stages, a.refiner_base).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)

    def lr_at(step):
        """Разгон 500 шагов, затем плавное снижение (косинус) до 2% к последнему шагу.
        Считается от номера шага, поэтому --resume с другим --steps работает."""
        warm = min(1.0, step / 500)
        cos = 0.5 * (1 + math.cos(math.pi * min(step, a.steps) / a.steps))
        return a.lr * warm * max(0.02, cos)
    start, best = 0, -1.0
    if a.resume and os.path.exists("checkpoints/last.pt"):
        ck = torch.load("checkpoints/last.pt", map_location=device)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start, best = ck["step"], ck.get("best", -1.0)
        print(f"Продолжаю с шага {start}")
    if a.init and not a.resume:
        ck = torch.load(a.init, map_location=device)
        if a.stages > 0 and ck.get("stages", 0) == 0:
            net.base_net.load_state_dict(ck["model"])     # обычная сеть -> первая часть сети с физикой
            print(f"Первая часть сети взята из {a.init}, физические блоки добавлены (stages={a.stages})")
        else:
            net.load_state_dict(ck["model"])
            print(f"Дообучение с весов {a.init}")
    print(f"Параметров в сети: {sum(p.numel() for p in net.parameters()) / 1e6:.1f} млн")

    log_new = not os.path.exists("train_log.csv") or not a.resume
    log = open("train_log.csv", "w" if log_new else "a", newline="")
    w = csv.writer(log)
    if log_new:
        w.writerow(["step", "loss", "val_KK", "val_SSIM", "sec"])
    t0, run = time.time(), []
    net.train()
    for step in range(start + 1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        T = sample(cache, a.batch, rng, device)
        loss = PT.loss_fn(net(T), T, w_ssim=a.w_ssim)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run.append(loss.item())
        if step % 100 == 0:
            print(f"шаг {step}/{a.steps}  ошибка {np.mean(run[-100:]):.3f}  {time.time() - t0:.0f} с", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            kk, ss = validate(net, val, device)
            score = kk          # лучшая — по КК: SSIM при нормировке на максимум «обманывается» яркой точкой
            mark = ""
            ck = {"model": net.state_dict(), "base": a.base, "stages": a.stages, "refiner_base": a.refiner_base,
                  "step": step, "val_KK": kk, "val_SSIM": ss}
            if score > best:
                best = score
                torch.save(ck, "checkpoints/best.pt")
                mark = "  ← лучшая, сохранена"
            torch.save({**ck, "opt": opt.state_dict(), "best": best}, "checkpoints/last.pt")
            print(f"  проверка: КК {kk:.3f}  SSIM {ss:.3f}{mark}", flush=True)
            w.writerow([step, np.mean(run[-100:]), kk, ss, int(time.time() - t0)]); log.flush()
    log.close()
    print("Готово. Лучшая модель: checkpoints/best.pt")


if __name__ == "__main__":
    main()
