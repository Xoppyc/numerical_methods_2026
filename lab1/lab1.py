import sys, requests, numpy as np, matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')

url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
try:
    res = requests.get(url, timeout=5).json()["results"]
except Exception:
    pts = [(48.164214,24.536044,1264.0),(48.164983,24.534836,1285.0),(48.165605,24.534068,1285.0),(48.166228,24.532915,1333.0),(48.166777,24.531927,1310.0),(48.167326,24.530884,1318.0),(48.167011,24.530061,1318.0),(48.166053,24.528039,1339.0),(48.166655,24.526064,1375.0),(48.166497,24.523574,1417.0),(48.166128,24.520214,1486.0),(48.165416,24.517170,1524.0),(48.164546,24.514640,1553.0),(48.163412,24.512980,1630.0),(48.162331,24.511715,1757.0),(48.162015,24.509462,1794.0),(48.162147,24.506932,1828.0),(48.161751,24.504244,1887.0),(48.161197,24.501793,1975.0),(48.160580,24.500537,1975.0),(48.160250,24.500106,2031.0)]
    res = [{'latitude': lat, 'longitude': lon, 'elevation': el} for lat, lon, el in pts]

n, coords, elev = len(res), [(p["latitude"], p["longitude"]) for p in res], [p["elevation"] for p in res]

def haversine(p1, p2):
    R, f1, f2 = 6371000, np.radians(p1[0]), np.radians(p2[0])
    df, dl = np.radians(p2[0] - p1[0]), np.radians(p2[1] - p1[1])
    a = np.sin(df / 2)**2 + np.cos(f1) * np.cos(f2) * np.sin(dl / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

dist = [0.0]
for i in range(1, n): dist.append(dist[-1] + haversine(coords[i-1], coords[i]))

with open("tabulation.txt", "w", encoding="utf-8") as f:
    f.write("№ | Latitude | Longitude | Distance (m) | Elevation (m)\n")
    for i in range(n): f.write(f"{i:2d}|{coords[i][0]:.6f}|{coords[i][1]:.6f}|{dist[i]:10.2f}|{elev[i]:8.2f}\n")

def cubic_spline(x, y):
    m, h = len(x) - 1, np.diff(x)
    alpha, beta, gamma = h[:-1], 2 * (h[:-1] + h[1:]), h[1:]
    delta = 3 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
    P, Q = np.zeros(m - 1), np.zeros(m - 1)
    P[0], Q[0] = -gamma[0] / beta[0], delta[0] / beta[0]
    for i in range(1, m - 1):
        d = beta[i] + alpha[i] * P[i-1]
        P[i], Q[i] = -gamma[i] / d, (delta[i] - alpha[i] * Q[i-1]) / d
    c = np.zeros(m + 1)
    c[m-1] = Q[-1]
    for i in range(m - 2, 0, -1): c[i] = P[i-1] * c[i+1] + Q[i-1]
    a, d = y[:-1], np.diff(c) / (3 * h)
    b = np.diff(y) / h - (2 * c[:-1] + c[1:]) * h / 3
    return a, b, c[:-1], d, alpha, beta, gamma, delta

def eval_spline(xe, x, a, b, c, d):
    i = np.clip(np.searchsorted(x, xe) - 1, 0, len(a) - 1)
    dx = xe - x[i]
    return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3

x_arr, y_arr = np.array(dist), np.array(elev)
a, b, c, d, alpha, beta, gamma, delta = cubic_spline(x_arr, y_arr)
print("alpha:", np.round(alpha, 1), "\nbeta :", np.round(beta, 1), "\ngamma:", np.round(gamma, 1))
print("c:", np.round(c, 5), "\na:", np.round(a, 1), "\nb:", np.round(b, 3), "\nd:", np.round(d, 7))

xx = np.linspace(x_arr[0], x_arr[-1], 500)
plt.figure(figsize=(9, 4.5))
for k in [10, 15, 20]:
    idx = np.round(np.linspace(0, n - 1, k)).astype(int)
    ak, bk, ck, dk, *_ = cubic_spline(x_arr[idx], y_arr[idx])
    err = np.max(np.abs(eval_spline(x_arr, x_arr[idx], ak, bk, ck, dk) - y_arr))
    print(f"Сплайн ({k} вузлів): макс. похибка = {err:.2f} м")
    plt.plot(xx, eval_spline(xx, x_arr[idx], ak, bk, ck, dk), label=f"{k} вузлів (Err: {err:.1f}м)")

plt.scatter(x_arr, y_arr, color='black', s=22, label='GPS-точки', zorder=5)
plt.title("Висотний профіль маршруту Заросляк - Говерла"), plt.xlabel("Відстань (м)"), plt.ylabel("Висота (м)")
plt.grid(True), plt.legend(), plt.tight_layout(), plt.savefig("profile_splines.png", dpi=200), plt.close()

ascent = sum(max(y_arr[i] - y_arr[i-1], 0) for i in range(1, n))
grad = np.gradient(eval_spline(xx, x_arr, a, b, c, d), xx) * 100
print(f"Довжина: {x_arr[-1]:.1f}м | Підйом: {ascent:.1f}м | Спуск: {sum(max(y_arr[i-1]-y_arr[i], 0) for i in range(1, n)):.1f}м")
print(f"Макс. підйом: {np.max(grad):.1f}% | Макс. спуск: {np.min(grad):.1f}% | Сер. градієнт: {np.mean(np.abs(grad)):.1f}%")
print(f"Робота (80 кг): {(80 * 9.81 * ascent)/1000:.1f} кДж | {(80 * 9.81 * ascent)/4184:.1f} ккал")
