"""Запуск решения для проверки (в т. ч. в Docker): комплекты по 4 изображения -> фазовый ДОЭ.

  python infer.py --input input --output output

Вход (--input):
  - папка, где лежат 4 изображения (один комплект), или
  - папка с подпапками, в каждой подпапке 4 изображения (несколько комплектов).
  Порядок изображений = порядок плоскостей 24,1 / 25,3 / 26,5 / 27,7 мм (по имени файла, по алфавиту).
  Форматы: png, jpg, bmp, tif. Размер и цвет приводятся к 128×128, оттенки серого.

Выход (--output), для каждого комплекта папка <имя комплекта>/:
  doe.png                 — фазовый ДОЭ, 128×128, 8 бит, grayscale, 0 = −π, 255 = +π
  restored_1..4.png       — восстановление Френелем (модуль |K|, нормировка на максимум)
  metrics.csv             — SSIM и Пирсон для каждой из 4 пар
И общий файл results.csv: средние SSIM, Пирсон и время синтеза по каждому комплекту + итог.

ДОЭ синтезирует только нейросеть за один проход. Время — чистое время работы сети на один комплект.
"""
import argparse
import glob
import os
import time
import numpy as np
import torch
from PIL import Image
from doe import physics_np as P
from doe.config import DISTANCES
from doe.device import pick_device
from doe.metrics_np import evaluate
from doe.model import load_model, default_model_path

EXT = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def list_images(folder):
    files = []
    for e in EXT:
        files += glob.glob(os.path.join(folder, e)) + glob.glob(os.path.join(folder, e.upper()))
    return sorted(set(files))


def find_sets(root):
    """Список (имя комплекта, [4 файла])."""
    own = list_images(root)
    if len(own) >= 4:
        return [(os.path.basename(os.path.normpath(root)) or "set", own[:4])]
    sets = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            imgs = list_images(p)
            if len(imgs) >= 4:
                if len(imgs) > 4:
                    print(f"  {d}: найдено {len(imgs)} изображений, беру первые 4 по алфавиту")
                sets.append((d, imgs[:4]))
            else:
                print(f"  {d}: меньше 4 изображений — пропускаю")
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="input")
    ap.add_argument("--output", default="output")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()

    model_path = a.model or default_model_path()
    dev = pick_device()
    net = load_model(model_path, dev)
    print(f"Модель: {model_path}")
    sets = find_sets(a.input)
    if not sets:
        raise SystemExit(f"В {a.input} не найдено ни одного комплекта из 4 изображений")
    os.makedirs(a.output, exist_ok=True)

    with torch.no_grad():                                   # прогрев (первый запуск всегда дольше)
        net(torch.zeros(1, 4, 128, 128, device=dev))

    rows_all = []
    for name, files in sets:
        targets = [P.load_gray(f) for f in files]
        T = torch.tensor(np.stack(targets)[None], dtype=torch.float32, device=dev)
        with torch.no_grad():
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            phase = net(T)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        phase = phase[0].cpu().numpy()

        out = os.path.join(a.output, name)
        os.makedirs(out, exist_ok=True)
        P.save_doe(phase, os.path.join(out, "doe.png"))
        q = P.uint8_to_phase(np.asarray(Image.open(os.path.join(out, "doe.png"))))   # как будет у проверяющих
        recs = [P.to_display(A) for A in P.reconstruct_all(q)]
        for k, r in enumerate(recs):
            P.save_gray(r, os.path.join(out, f"restored_{k + 1}.png"))
        recs = [P.load_gray(os.path.join(out, f"restored_{k + 1}.png")) for k in range(4)]
        rows = evaluate(targets, recs)
        with open(os.path.join(out, "metrics.csv"), "w", encoding="utf-8") as f:
            f.write("image;file;z_mm;SSIM;Pearson\n")
            for r, fn, z in zip(rows, files, DISTANCES):
                f.write(f"{r['img']};{os.path.basename(fn)};{z * 1e3:.1f};{r['SSIM']:.4f};{r['KK']:.4f}\n")
        ssim = float(np.mean([r["SSIM"] for r in rows]))
        kk = float(np.mean([r["KK"] for r in rows]))
        rows_all.append((name, ssim, kk, dt * 1000))
        print(f"  {name:<20} SSIM {ssim:.4f}  Pearson {kk:.4f}  время {dt * 1000:.1f} мс")

    with open(os.path.join(a.output, "results.csv"), "w", encoding="utf-8") as f:
        f.write("set;SSIM;Pearson;time_ms\n")
        for n, s, k, t in rows_all:
            f.write(f"{n};{s:.4f};{k:.4f};{t:.2f}\n")
        f.write(f"MEAN;{np.mean([r[1] for r in rows_all]):.4f};{np.mean([r[2] for r in rows_all]):.4f};"
                f"{np.mean([r[3] for r in rows_all]):.2f}\n")
    print(f"Итог по {len(rows_all)} комплектам: SSIM {np.mean([r[1] for r in rows_all]):.4f}  "
          f"Pearson {np.mean([r[2] for r in rows_all]):.4f}  время {np.mean([r[3] for r in rows_all]):.1f} мс")
    print(f"Результаты: {a.output}")


if __name__ == "__main__":
    main()
