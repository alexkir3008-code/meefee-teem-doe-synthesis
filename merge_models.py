"""Объединение двух моделей, дообученных из ОДНОЙ стартовой модели: усреднение весов («model soup»,
Wortsman et al., 2022). Получается одна модель — та же скорость синтеза.

  python merge_models.py models/best_tikhon_night.pt models/best_radomir_night.pt

Проверяет: модель A, модель B и смеси весов (A·(1−t) + B·t) для t = 0.25, 0.5, 0.75
на 24 фиксированных проверочных комплектах (не тестовых!) по метрикам проверяющих.
Лучшая по среднему гармоническому Pearson и SSIM сохраняется в models/best_merged.pt.
Если лучше всех оказалась одна из исходных моделей — так и будет написано.
"""
import argparse
import numpy as np
import torch
from doe import physics_np as P
from doe.data import ImageSource
from doe.device import pick_device
from doe.metrics_np import evaluate
from doe.model import build_model


def score(net, sets, dev):
    kk, ss = [], []
    with torch.no_grad():
        ph = net(torch.tensor(sets, device=dev)).cpu().numpy()
    for T, p in zip(sets, ph):
        q = P.uint8_to_phase(P.phase_to_uint8(p))
        R = [np.round(P.to_display(A) * 255) / 255 for A in P.reconstruct_all(q)]
        rows = evaluate(list(T.astype(np.float64)), R)
        kk += [r["KK"] for r in rows]
        ss += [r["SSIM"] for r in rows]
    k, s = float(np.mean(kk)), float(np.mean(ss))
    return k, s, 2 * k * s / (k + s + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out", default="models/best_merged.pt")
    a = ap.parse_args()
    dev = pick_device()
    ca = torch.load(a.a, map_location="cpu")
    cb = torch.load(a.b, map_location="cpu")
    for key in ("arch", "base", "stages", "refiner_base"):
        if ca.get(key) != cb.get(key):
            raise SystemExit(f"Модели разной архитектуры ({key}: {ca.get(key)} и {cb.get(key)}) — объединять нельзя")
    net = build_model(ca.get("base", 32), ca.get("stages", 0), ca.get("refiner_base", 24), ca.get("arch", "unet")).to(dev)
    net.eval()

    src = ImageSource(seed=2024)
    sets = np.stack([src.quad() for _ in range(24)]).astype(np.float32)

    results = []
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        sd = {k: (ca["model"][k].float() * (1 - t) + cb["model"][k].float() * t).to(ca["model"][k].dtype)
              if ca["model"][k].is_floating_point() else ca["model"][k] for k in ca["model"]}
        net.load_state_dict(sd)
        k, s, h = score(net, sets, dev)
        name = "только A" if t == 0 else "только B" if t == 1 else f"смесь t={t}"
        results.append((h, t, name, k, s, sd))
        print(f"{name:<14} Pearson {k:.4f}  SSIM {s:.4f}  баланс {h:.4f}", flush=True)

    h, t, name, k, s, sd = max(results, key=lambda r: r[0])
    out = {**ca, "model": sd, "val_KK": k, "val_SSIM": s, "merged_from": [a.a, a.b], "merge_t": t}
    torch.save(out, a.out)
    print(f"\nЛучший вариант: {name} (Pearson {k:.4f}, SSIM {s:.4f}) -> сохранён в {a.out}")
    print("Проверьте глазами: python solve.py --dir examples --model " + a.out)


if __name__ == "__main__":
    main()
