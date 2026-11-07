"""ГЛАВНЫЙ СКРИПТ: 4 изображения -> ДОЭ + 4 восстановленных изображения + метрики (формат кейса).

  python solve.py img1.png img2.png img3.png img4.png
  python solve.py --dir input_images            # 4 картинки из папки (по алфавиту)

Порядок картинок = порядок плоскостей: 24,1 / 25,3 / 26,5 / 27,7 мм.
Результат: папка results/<время>/
  final/doe.png                — ДОЭ, 128×128, 8 бит, 0 = −π, 255 = +π
  final/restored_1..4.png      — восстановленные изображения, 128×128, 8 бит
  final/metrics.csv, .json     — КК и ИСС для каждой пары
  summary.png                  — картинка для презентации
  methods.csv                  — сравнение всех методов (время и метрики)
Все метрики считаются по сохранённому doe.png (уже после перевода в 8 бит).
"""
import argparse
import glob
import json
import os
import shutil
import time
import numpy as np
from doe import physics_np as P
from doe.config import DISTANCES
from doe.metrics_np import evaluate

try:
    import torch
    from doe.device import pick_device
    from doe import physics_torch as PT
    from doe.model import load_model
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def score_rows(rows):
    return float(np.mean([(r["KK"] + r["SSIM"]) / 2 for r in rows]))


def finalize(phase, targets, folder):
    """Сохранить ДОЭ, восстановить из СОХРАНЁННОГО png и посчитать метрики."""
    os.makedirs(folder, exist_ok=True)
    P.save_doe(phase, os.path.join(folder, "doe.png"))
    from PIL import Image
    q = P.uint8_to_phase(np.asarray(Image.open(os.path.join(folder, "doe.png"))))
    recs = [P.to_display(I, t) for I, t in zip(P.reconstruct_all(q), targets)]
    for k, r in enumerate(recs):
        P.save_gray(r, os.path.join(folder, f"restored_{k + 1}.png"))
    recs = [P.load_gray(os.path.join(folder, f"restored_{k + 1}.png")) for k in range(4)]
    rows = evaluate(targets, recs)
    for r, z in zip(rows, DISTANCES):
        r["z_mm"] = round(z * 1e3, 1)
    with open(os.path.join(folder, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(folder, "metrics.csv"), "w", encoding="utf-8") as f:
        f.write("изображение;z_мм;КК_Пирсон;ИСС_SSIM;ИСС_по_формуле_слайда\n")
        for r in rows:
            f.write(f"{r['img']};{r['z_mm']};{r['KK']:.4f};{r['SSIM']:.4f};{r['ISS_slide']:.4f}\n")
        f.write(f"среднее;;{np.mean([r['KK'] for r in rows]):.4f};{np.mean([r['SSIM'] for r in rows]):.4f};{np.mean([r['ISS_slide'] for r in rows]):.4f}\n")
    return rows, recs, q


def summary_figure(targets, recs, phase, rows, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 5, figsize=(15, 6.4))
    ax[0, 0].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi); ax[0, 0].set_title("ДОЭ (фаза)")
    ax[1, 0].axis("off")
    ax[1, 0].text(0, 0.5, title, fontsize=11, va="center")
    for k in range(4):
        ax[0, k + 1].imshow(targets[k], cmap="gray", vmin=0, vmax=1)
        ax[0, k + 1].set_title(f"Исходное {k + 1}  (z = {DISTANCES[k] * 1e3:.1f} мм)")
        ax[1, k + 1].imshow(recs[k], cmap="gray", vmin=0, vmax=1)
        ax[1, k + 1].set_title(f"КК {rows[k]['KK']:.3f}   ИСС {rows[k]['SSIM']:.3f}")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--dir")
    ap.add_argument("--model", nargs="+", default=None,
                    help="одна или несколько моделей (ансамбль): --model checkpoints/best.pt best_radomir.pt")
    ap.add_argument("--refine-steps", type=int, default=500)
    ap.add_argument("--starts", type=int, default=4, help="сколько вариантов доводить параллельно")
    ap.add_argument("--w-ssim", type=float, default=0.5, help="вес ИСС в доводке (меньше — выше КК)")
    ap.add_argument("--gs-iters", type=int, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--best-any", action="store_true", help="в final — лучший из всех методов, а не только нейросетевых")
    a = ap.parse_args()
    if a.model is None:
        from doe.model import default_model_path
        a.model = [default_model_path()]

    paths = a.images or sorted(sum([glob.glob(os.path.join(a.dir, e)) for e in ("*.png", "*.jpg", "*.bmp")], []))
    assert len(paths) == 4, f"Нужно ровно 4 изображения, найдено {len(paths)}"
    targets = [P.load_gray(p) for p in paths]
    out = a.out or os.path.join("results", time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(os.path.join(out, "input"), exist_ok=True)
    for k, p in enumerate(paths):
        shutil.copy(p, os.path.join(out, "input", f"{k + 1}_{os.path.basename(p)}"))
    print("Картинки по плоскостям:", *[f"{z * 1e3:.1f} мм ← {os.path.basename(p)}" for z, p in zip(DISTANCES, paths)], sep="\n  ")

    cands = {}                                     # метод -> (фаза, секунды)
    t0 = time.time(); cands["Гершберг–Сакстон"] = (P.gs_multiplane(targets, a.gs_iters), time.time() - t0)

    if HAS_TORCH:
        dev = pick_device()
        T = torch.tensor(np.stack(targets)[None], dtype=torch.float32, device=dev)
        inits, start_names = [], []
        models = [m for m in a.model if os.path.exists(m)]
        for m in a.model:
            if not os.path.exists(m):
                print(f"Модель {m} не найдена — пропускаю.")
        for mi, m in enumerate(models):
            net = load_model(m, dev)
            with torch.no_grad():
                net(T)                                           # прогрев
                if dev.type == "cuda": torch.cuda.synchronize()
                t0 = time.time(); ph_nn = net(T)
                if dev.type == "cuda": torch.cuda.synchronize()
                dt = time.time() - t0
            label = "Нейросеть" if len(models) == 1 else f"Нейросеть {os.path.basename(m)}"
            cands[label] = (ph_nn[0].cpu().numpy(), dt)
            inits.append(ph_nn[0]); start_names.append(f"{label} + доводка")
        if not models:
            print("Ни одной модели нет — сначала python train.py. Пока без нейросети.")
        inits.append(torch.tensor(cands["Гершберг–Сакстон"][0], dtype=torch.float32, device=dev))   # старт с классики — тянет КК вверх
        start_names.append("Гершберг–Сакстон + доводка")
        g = torch.Generator(device="cpu").manual_seed(0)
        while len(inits) < a.starts:                              # дополнительные старты
            base = inits[0] if inits else torch.tensor(cands["Гершберг–Сакстон"][0], dtype=torch.float32, device=dev)
            noise = (torch.rand(base.shape, generator=g) - 0.5).to(dev) * 0.6 * len(inits)
            inits.append(base + noise); start_names.append(f"{start_names[0]} (вариант {len(inits)})")
        init = torch.stack(inits)
        t0 = time.time()
        ref = PT.refine(T.expand(len(inits), -1, -1, -1), init, steps=a.refine_steps, w_ssim=a.w_ssim)
        if dev.type == "cuda": torch.cuda.synchronize()
        dt = time.time() - t0
        for nm, ph in zip(start_names, ref.cpu().numpy()):
            cands[nm] = (ph, dt)
    else:
        from doe.optimize_np import optimize
        print("PyTorch не установлен — использую numpy-оптимизацию (медленнее).")
        t0 = time.time(); cands["Градиентная оптимизация (numpy)"] = (optimize(targets, init=cands["Гершберг–Сакстон"][0], steps=300), time.time() - t0)

    results = {}
    for name, (ph, dt) in cands.items():
        rows, recs, q = finalize(ph, targets, os.path.join(out, "all_methods", name.replace(" ", "_").replace("+", "plus").replace("#", "n")))
        results[name] = (rows, recs, q, dt)
        print(f"  {name:<32} КК {np.mean([r['KK'] for r in rows]):.3f}  ИСС {np.mean([r['SSIM'] for r in rows]):.3f}  время {dt:.2f} с")

    with open(os.path.join(out, "methods.csv"), "w", encoding="utf-8") as f:
        f.write("метод;КК_ср;КК_мин;ИСС_ср;ИСС_мин;ИСС_слайд_ср;время_с\n")
        for name, (rows, _, _, dt) in results.items():
            kk = [r["KK"] for r in rows]; ss = [r["SSIM"] for r in rows]; sg = [r["ISS_slide"] for r in rows]
            f.write(f"{name};{np.mean(kk):.4f};{min(kk):.4f};{np.mean(ss):.4f};{min(ss):.4f};{np.mean(sg):.4f};{dt:.3f}\n")

    nn_names = [n for n in results if n.startswith("Нейросеть")]
    pool = list(results) if a.best_any or not nn_names else nn_names
    best = max(pool, key=lambda n: score_rows(results[n][0]))
    overall = max(results, key=lambda n: score_rows(results[n][0]))
    if overall != best:
        print(f"Примечание: по сумме КК+ИСС лучший вообще — «{overall}». Чтобы сдать его: --best-any")
    rows, recs, q, dt = results[best]
    final = os.path.join(out, "final")
    rows, recs, q = finalize(q, targets, final)
    summary_figure(targets, recs, q, rows,
                   f"{best}\nКК ср. {np.mean([r['KK'] for r in rows]):.3f}\nИСС ср. {np.mean([r['SSIM'] for r in rows]):.3f}\nвремя {dt:.2f} с",
                   os.path.join(out, "summary.png"))
    print(f"\nИтог: {best}\nКК: {[round(r['KK'], 3) for r in rows]}\nИСС: {[round(r['SSIM'], 3) for r in rows]}")
    print(f"Файлы для сдачи: {final}")


if __name__ == "__main__":
    main()
