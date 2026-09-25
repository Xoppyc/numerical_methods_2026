import os, sys, csv
import numpy as np
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
DIR = os.path.dirname(os.path.abspath(__file__))

def read_data(filename):
    with open(filename, 'r', newline='') as f:
        r = csv.DictReader(f)
        pts = [(float(row['Month']), float(row['Temp'])) for row in r]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])

x, y = read_data(os.path.join(DIR, "data.csv"))
n = len(x)

def form_system(x, y, m):
    A = np.array([[np.sum(x**(i + j)) for j in range(m + 1)] for i in range(m + 1)])
    b = np.array([np.sum(y * (x**i)) for i in range(m + 1)])
    return A, b

def gauss_solve(A_in, b_in):
    A, b = A_in.copy().astype(float), b_in.copy().astype(float)
    k_len = len(b)
    for k in range(k_len):
        max_r = k + np.argmax(np.abs(A[k:, k]))
        if max_r != k:
            A[[k, max_r]] = A[[max_r, k]]
            b[[k, max_r]] = b[[max_r, k]]
        for i in range(k + 1, k_len):
            f = A[i, k] / A[k, k]
            A[i, k:] -= f * A[k, k:]
            b[i] -= f * b[k]
    x_sol = np.zeros(k_len)
    for i in range(k_len - 1, -1, -1):
        x_sol[i] = (b[i] - np.dot(A[i, i + 1:], x_sol[i + 1:])) / A[i, i]
    return x_sol

def eval_poly(xe, c):
    return sum(ci * (xe**i) for i, ci in enumerate(c))

models = {}
print("Дослідження дисперсії для степенів m = 1..4:")
for m in range(1, 5):
    A, b_vec = form_system(x, y, m)
    c = gauss_solve(A, b_vec)
    y_pred = eval_poly(x, c)
    var = np.mean((y - y_pred)**2)
    models[m] = {'c': c, 'y_pred': y_pred, 'var': var}
    print(f"  m = {m}: дисперсія = {var:6.2f}, RMSE = {np.sqrt(var):5.2f} °C")

opt_m = min(models, key=lambda k: models[k]['var'])
opt_c = models[opt_m]['c']
print(f"\nОптимальний степінь: m = {opt_m} (коефіцієнти: {np.round(opt_c, 4)})")

x_fut = np.array([25, 26, 27])
y_fut = eval_poly(x_fut, opt_c)
print(f"Прогноз температури на наступні 3 місяці: {np.round(y_fut, 1)} °C")

with open(os.path.join(DIR, "tabulation.txt"), "w", encoding="utf-8") as f:
    f.write("Місяць | Факт | Апрокс (m=4) | Похибка\n")
    for xi, yi, ypi in zip(x, y, models[opt_m]['y_pred']):
        f.write(f"{int(xi):6d} | {yi:5.1f} | {ypi:12.2f} | {yi - ypi:8.2f}\n")

xx = np.linspace(1, 27, 300)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

ax1.scatter(x, y, color='black', s=25, label='Фактичні дані (24 міс.)', zorder=5)
for m, style in [(1, ':'), (2, '--'), (4, '-')]:
    ax1.plot(xx, eval_poly(xx, models[m]['c']), label=f'm={m} (var={models[m]["var"]:.1f})', ls=style)
ax1.scatter(x_fut, y_fut, color='red', marker='*', s=60, label='Прогноз (3 міс.)', zorder=6)
ax1.set_title("Апроксимація температурних даних методом найменших квадратів")
ax1.set_ylabel("Температура (°C)"), ax1.grid(True), ax1.legend(), ax1.set_ylim(-25, 30)

ax2.bar(x, y - models[opt_m]['y_pred'], color='#1f77b4', alpha=0.7, label='Похибка ε(x) для m=4')
ax2.axhline(0, color='black', lw=0.8)
ax2.set_xlabel("Місяць"), ax2.set_ylabel("Похибка (°C)"), ax2.grid(True), ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(DIR, "plot_mnk.png"), dpi=200)
plt.close()
print(f"[OK] Графік збережено: {os.path.join(DIR, 'plot_mnk.png')}")
