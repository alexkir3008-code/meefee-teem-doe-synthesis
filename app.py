"""Интерфейс в браузере: загрузите 4 изображения -> ДОЭ, восстановленные изображения, метрики, архив для сдачи.
  python app.py                              (откроется http://127.0.0.1:7860)
  python app.py --model models/best.pt       (выбрать модель)"""
import argparse
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import gradio as gr
from PIL import Image
from doe import physics_np as P
from doe.config import DISTANCES
from doe.metrics_np import evaluate

try:
    import torch
    from doe.device import pick_device
    from doe import physics_torch as PT
    from doe.model import load_model, default_model_path
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--model", default=None)
    MODEL = _ap.parse_known_args()[0].model or default_model_path()
    DEV = pick_device()
    NET = load_model(MODEL, DEV) if os.path.exists(MODEL) else None
    # Сеть всегда работает в одном и том же потоке: видеокарта настраивает свои библиотеки
    # отдельно для каждого потока, и первый запуск в новом потоке длится десятки секунд.
    POOL = ThreadPoolExecutor(max_workers=1)

    def _infer(T):
        with torch.no_grad():
            T = torch.as_tensor(T, dtype=torch.float32, device=DEV)
            if DEV.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            ph = NET(T)
            if DEV.type == "cuda":
                torch.cuda.synchronize()
            return ph[0].cpu().numpy(), time.time() - t0

    if NET is not None:
        print(f"Модель: {MODEL}")
        for _ in range(3):                         # прогрев в том же потоке, где потом считается ДОЭ
            POOL.submit(_infer, np.zeros((1, 4, 128, 128), np.float32)).result()
        print("Сеть прогрета, можно открывать http://127.0.0.1:7860")
except Exception:
    torch = None
    NET = None

METHODS = (["Нейросеть"] if NET is not None else []) + ["Гершберг–Сакстон (для сравнения)"]


def prep(img):
    im = Image.fromarray(img).convert("L").resize((128, 128), Image.BICUBIC)
    return np.asarray(im, dtype=np.float64) / 255


def run(i1, i2, i3, i4, method):
    imgs = [i1, i2, i3, i4]
    if any(i is None for i in imgs):
        raise gr.Error("Загрузите все 4 изображения")
    T = [prep(i) for i in imgs]
    t0 = time.time()
    if method.startswith("Гершберг"):
        phase = P.gs_multiplane(T, 300)
        dt = time.time() - t0
    else:
        phase, dt = POOL.submit(_infer, np.stack(T)[None].astype(np.float32)).result()
    out = tempfile.mkdtemp()
    P.save_doe(phase, os.path.join(out, "doe.png"))
    q = P.uint8_to_phase(np.asarray(Image.open(os.path.join(out, "doe.png"))))
    R = [P.to_display(I, t) for I, t in zip(P.reconstruct_all(q), T)]
    for k, r in enumerate(R):
        P.save_gray(r, os.path.join(out, f"restored_{k + 1}.png"))
    R = [P.load_gray(os.path.join(out, f"restored_{k + 1}.png")) for k in range(4)]
    rows = evaluate(T, R)
    table = [[r["img"], f"{DISTANCES[k] * 1e3:.1f}", round(r["KK"], 4), round(r["SSIM"], 4)] for k, r in enumerate(rows)]
    table.append(["среднее", "", round(np.mean([r["KK"] for r in rows]), 4), round(np.mean([r["SSIM"] for r in rows]), 4)])
    zip_path = shutil.make_archive(os.path.join(tempfile.mkdtemp(), "doe_result"), "zip", out)
    to_u8 = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return (np.asarray(Image.open(os.path.join(out, "doe.png"))), *[to_u8(r) for r in R], table,
            f"Метод: {method}. Время: {dt:.2f} с", zip_path)


with gr.Blocks(title="Синтез ДОЭ") as demo:
    gr.Markdown("## Нейросетевой синтез ДОЭ: 4 изображения на 4 расстояниях\n532 нм · пиксель 10 мкм · 24,1 / 25,3 / 26,5 / 27,7 мм")
    with gr.Row():
        ins = [gr.Image(label=f"Изображение {k + 1} · {DISTANCES[k] * 1e3:.1f} мм", height=180) for k in range(4)]
    with gr.Row():
        method = gr.Radio(METHODS, value=METHODS[0], label="Метод")
        btn = gr.Button("Рассчитать ДОЭ", variant="primary")
    with gr.Row():
        doe = gr.Image(label="ДОЭ (0 = −π, 255 = +π)", height=180)
        outs = [gr.Image(label=f"Восстановлено {k + 1}", height=180) for k in range(4)]
    table = gr.Dataframe(headers=["№", "z, мм", "КК", "ИСС"], label="Метрики")
    info = gr.Markdown()
    file = gr.File(label="Скачать файлы для сдачи (zip)")
    btn.click(run, [*ins, method], [doe, *outs, table, info, file])

if __name__ == "__main__":
    demo.launch()
