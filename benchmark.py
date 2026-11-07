"""Сравнение методов на многих наборах по 4 картинки — таблица и график для презентации.
  python benchmark.py --sets 20
Результат: benchmark.csv, benchmark.png"""
import argparse
import time
import numpy as np
import torch
from doe import physics_np as P
from doe import physics_torch as PT
from doe.data import ImageSource
from doe.device import pick_device
from doe.metrics_np import evaluate
from doe.model import load_model, default_model_path


def measure(phase, T):
    q = P.uint8_to_phase(P.phase_to_uint8(phase))
    R = [P.to_display(I, t) for I, t in zip(P.reconstruct_all(q), T)]
    R = [np.round(r * 255) / 255 for r in R]
    rows = evaluate(list(T), R)
    return [r["KK"] for r in rows], [r["SSIM"] for r in rows]


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=int, default=20)
    ap.add_argument("--model", default=None)
    ap.add_argument("--refine-steps", type=int, default=200)
    a = ap.parse_args()
    a.model = a.model or default_model_path()
    dev = pick_device()
    net = load_model(a.model, dev)
    src = ImageSource(seed=777)
    sets = [src.quad().astype(np.float32) for _ in range(a.sets)]
    res = {m: {"kk": [], "ss": [], "t": []} for m in ["Гершберг–Сакстон (300 ит.)", "Нейросеть", "Нейросеть + доводка"]}
    with torch.no_grad():
        net(torch.tensor(sets[0][None], device=dev))
    for i, T in enumerate(sets):
        t0 = time.time(); ph = P.gs_multiplane(list(T.astype(np.float64)), 300); dt = time.time() - t0
        kk, ss = measure(ph, T); r = res["Гершберг–Сакстон (300 ит.)"]; r["kk"] += kk; r["ss"] += ss; r["t"].append(dt)
        Tt = torch.tensor(T[None], device=dev)
        with torch.no_grad():
            sync(dev); t0 = time.time(); phn = net(Tt); sync(dev); dt_nn = time.time() - t0
        kk, ss = measure(phn[0].cpu().numpy(), T); r = res["Нейросеть"]; r["kk"] += kk; r["ss"] += ss; r["t"].append(dt_nn)
        sync(dev); t0 = time.time(); phr = PT.refine(Tt, phn, steps=a.refine_steps); sync(dev); dt = time.time() - t0 + dt_nn
        kk, ss = measure(phr[0].cpu().numpy(), T); r = res["Нейросеть + доводка"]; r["kk"] += kk; r["ss"] += ss; r["t"].append(dt)
        print(f"набор {i + 1}/{a.sets}", flush=True)
    with open("benchmark.csv", "w", encoding="utf-8") as f:
        f.write("метод;КК_ср;КК_мин;ИСС_ср;ИСС_мин;время_мс\n")
        print(f"\n{'метод':<28}{'КК':>7}{'ИСС':>7}{'время, мс':>11}")
        for m, r in res.items():
            f.write(f"{m};{np.mean(r['kk']):.4f};{np.min(r['kk']):.4f};{np.mean(r['ss']):.4f};{np.min(r['ss']):.4f};{np.mean(r['t']) * 1000:.1f}\n")
            print(f"{m:<28}{np.mean(r['kk']):7.3f}{np.mean(r['ss']):7.3f}{np.mean(r['t']) * 1000:11.1f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(res)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(names))
    ax[0].bar(x - 0.18, [np.mean(res[n]["kk"]) for n in names], 0.36, label="КК")
    ax[0].bar(x + 0.18, [np.mean(res[n]["ss"]) for n in names], 0.36, label="ИСС")
    ax[0].set_xticks(x, names, fontsize=8); ax[0].set_ylim(0, 1); ax[0].legend(); ax[0].set_title("Качество (больше — лучше)")
    ax[1].bar(x, [np.mean(res[n]["t"]) * 1000 for n in names], color="gray")
    ax[1].set_xticks(x, names, fontsize=8); ax[1].set_yscale("log"); ax[1].set_title("Время расчёта, мс (меньше — лучше)")
    plt.tight_layout(); plt.savefig("benchmark.png", dpi=130)
    print("Готово: benchmark.csv, benchmark.png")


if __name__ == "__main__":
    main()
