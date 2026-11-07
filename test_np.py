import time, numpy as np
from doe.config import DISTANCES
from doe import physics_np as P
from doe.metrics_np import evaluate, pearson, ssim

rng = np.random.default_rng(0)
# 1) полная формула и упрощённая дают одну и ту же форму изображения
ph = rng.uniform(-np.pi, np.pi, (128, 128))
for k in range(4):
    a = np.abs(P.fresnel_field(ph, k)) ** 2; b = P.intensity(ph, k)
    assert np.allclose(a / a.sum(), b / b.sum()), k
print("Формула слайда == упрощённый расчёт: OK")
# 2) формат png туда-обратно: ошибка фазы не больше половины шага квантования
q = P.uint8_to_phase(P.phase_to_uint8(ph))
err = np.abs(np.angle(np.exp(1j * (q - ph)))).max()
assert err <= np.pi / 255 + 1e-9, err
print(f"Квантование 8 бит: макс. ошибка фазы {err:.4f} рад: OK")
# 3) метрики: одинаковые картинки -> 1
img = rng.random((128, 128))
r = evaluate([img], [img])[0]
assert abs(r["KK"] - 1) < 1e-9 and abs(r["SSIM"] - 1) < 1e-9 and abs(r["ISS_slide"] - 1) < 1e-9
print("Метрики на одинаковых картинках = 1: OK")
for z in DISTANCES:
    print(f"  z = {z*1e3:.1f} мм, шаг в плоскости восстановления {P.out_pixel(z)*1e6:.2f} мкм")
