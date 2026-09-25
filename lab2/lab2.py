import os, sys, csv, math
import numpy as np
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
DIR = os.path.dirname(os.path.abspath(__file__))

def read_data(filename):
    with open(filename, 'r', newline='') as f:
        r = csv.DictReader(f)
        pts = [(float(row['n']), float(row['t'])) for row in r]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])

x, y = read_data(os.path.join(DIR, "data.csv"))
n = len(x)

def divided_diff(x, y):
    m = len(x)
    t = np.zeros((m, m))
    t[:, 0] = y
    for j in range(1, m):
        for i in range(m - j):
            t[i, j] = (t[i+1, j-1] - t[i, j-1]) / (x[i+j] - x[i])
    return t

def eval_newton(xe, x, coeffs):
    res = coeffs[-1]
    for i in range(len(coeffs) - 2, -1, -1):
        res = res * (xe - x[i]) + coeffs[i]
    return res

dd_table = divided_diff(x, y)
c_newton = dd_table[0, :]

fd = np.zeros((n, n))
fd[:, 0] = y
for j in range(1, n):
    for i in range(n - j): fd[i, j] = fd[i+1, j-1] - fd[i, j-1]

def eval_factorial(q, deltas):
    res, term = 0.0, 1.0
    for j in range(len(deltas)):
        if j > 0: term *= (q - (j - 1))
        res += (deltas[j] / math.factorial(j)) * term
    return res

def eval_lagrange(xe, x, y):
    return sum(y[i] * np.prod([(xe - x[j]) / (x[i] - x[j]) for j in range(len(x)) if i != j]) for i in range(len(x)))

t_newton = eval_newton(6000, x, c_newton)
t_fact   = eval_factorial(np.log2(6000 / 1000), fd[0, :])
t_lagr   = eval_lagrange(6000, x, y)

print("Таблиця розділених різниць:\n", np.round(dd_table, 6))
print(f"\nПрогноз часу для n = 6000:\n  Ньютон:      {t_newton:.2f} мс\n  Факторіали:  {t_fact:.2f} мс\n  Лагранж:     {t_lagr:.2f} мс")

with open(os.path.join(DIR, "tabulation.txt"), "w", encoding="utf-8") as f:
    f.write("n | t_exp | P_Newton(n)\n" + "\n".join(f"{int(xi):5d} | {yi:5.1f} | {eval_newton(xi, x, c_newton):5.1f}" for xi, yi in zip(x, y)))

xx = np.linspace(1000, 16000, 400)
plt.figure(figsize=(9, 5))
plt.scatter(x, y, color='red', s=45, zorder=5, label='Експериментальні дані (5 точок)')
plt.scatter([6000], [t_newton], color='blue', s=70, marker='*', zorder=6, label=f'Прогноз P(6000) = {t_newton:.1f} мс')
plt.plot(xx, [eval_newton(xi, x, c_newton) for xi in xx], 'k-', lw=1.8, label='Ньютон (5 вузлів, степінь 4)')

f_ref = lambda v: eval_newton(v, x, c_newton)
for N, col, ls in [(10, 'green', '--'), (20, 'purple', ':')]:
    x_n = np.linspace(1000, 16000, N)
    c_n = divided_diff(x_n, [f_ref(xi) for xi in x_n])[0, :]
    y_curve = [eval_newton(xi, x_n, c_n) for xi in xx]
    plt.plot(xx, y_curve, color=col, linestyle=ls, label=f'{N} вузлів (степінь {N-1})')
    print(f"N = {N:2d} вузлів: P(6000) = {eval_newton(6000, x_n, c_n):.2f} мс")

plt.title("Інтерполяція залежності t(n) та дослідження кількості вузлів")
plt.xlabel("Розмір вхідних даних n"), plt.ylabel("Час виконання t (мс)")
plt.ylim(-10, 110), plt.grid(True), plt.legend(), plt.tight_layout()
plt.savefig(os.path.join(DIR, "plot_interpolation.png"), dpi=200)
plt.close()
print(f"[OK] Графік збережено: {os.path.join(DIR, 'plot_interpolation.png')}")
