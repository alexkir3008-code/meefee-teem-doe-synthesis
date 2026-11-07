"""Интерфейс в браузере: загрузите 4 изображения -> ДОЭ, восстановленные изображения, метрики, архив для сдачи.
  python app.py   (откроется http://127.0.0.1:7860)"""
import os
import shutil
import tempfile
import time
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
    DEV = pick_device()
    NET = load_model(default_model_path(), DEV) if os.path.exists(default_model_path()) else None
except Exception:
    torch = None
    NET = None

METHODS = (["Нейросеть + доводка", "Нейросеть"] if NET is not None else []) \
    + (["Гершберг–Сакстон + доводка"] if torch is not None else []) + ["Гершберг–Сакстон"]


def prep(img):
    im = Image.fromarray(img).convert("L").resize((128, 128), Image.BICUBIC)
    return np.asarray(im, dtype=np.float64) / 255


def run(i1, i2, i3, i4, method, steps):
    imgs = [i1, i2, i3, i4]
    if any(i is None for i in imgs):
        raise gr.Error("Загрузите все 4 изображения")
    T = [prep(i) for i in imgs]
    t0 = time.time()
    if method == "Гершберг–Сакстон":
        phase = P.gs_multiplane(T, 300)
    elif method == "Гершберг–Сакстон + доводка":
        Tt = torch.tensor(np.stack(T)[None], dtype=torch.float32, device=DEV)
        init = torch.tensor(P.gs_multiplane(T, 150)[None], dtype=torch.float32, device=DEV)
        phase = PT.refine(Tt, init, steps=int(steps))[0].cpu().numpy()
    else:
        Tt = torch.tensor(np.stack(T)[None], dtype=torch.float32, device=DEV)
        with torch.no_grad():
            ph = NET(Tt)
        if method == "Нейросеть + доводка":
            ph = PT.refine(Tt, ph, steps=int(steps))
        phase = ph[0].cpu().numpy()
    dt = time.time() - t0
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
        steps = gr.Slider(50, 1000, value=300, step=50, label="Шаги доводки")
        btn = gr.Button("Рассчитать ДОЭ", variant="primary")
    with gr.Row():
        doe = gr.Image(label="ДОЭ (0 = −π, 255 = +π)", height=180)
        outs = [gr.Image(label=f"Восстановлено {k + 1}", height=180) for k in range(4)]
    table = gr.Dataframe(headers=["№", "z, мм", "КК", "ИСС"], label="Метрики")
    info = gr.Markdown()
    file = gr.File(label="Скачать файлы для сдачи (zip)")
    btn.click(run, [*ins, method, steps], [doe, *outs, table, info, file])

if __name__ == "__main__":
    demo.launch()
