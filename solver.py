import json
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq
from scipy.special import erfc


# -1/2 psi'' + V psi = E psi, hbar = m = 1, на краях psi = 0
def solve(V, a, b, N, k):
    h = (b - a) / (N + 1)
    x = a + h * np.arange(1, N + 1)
    diag = 1.0 / h**2 + V(x)
    off = np.full(N - 1, -0.5 / h**2)
    E, psi = eigh_tridiagonal(diag, off, select="i", select_range=(0, k - 1))
    psi /= np.sqrt(h)  # чтобы интеграл |psi|^2 dx был 1
    for n in range(k):
        # знак произвольный, делаю первый горб вверх
        j = np.argmax(np.abs(psi[:, n]) > 1e-3 * np.abs(psi[:, n]).max())
        if psi[j, n] < 0:
            psi[:, n] *= -1
    return x, E, psi


def forbidden_probability(x, V, E, psi):
    # граница V = E между узлами, поэтому считаю на сетке в 20 раз мельче
    xf = np.linspace(x[0], x[-1], 20 * (len(x) - 1) + 1)
    pf = np.interp(xf, x, psi)
    return float(np.trapezoid(np.where(V(xf) > E, pf**2, 0.0), xf))


def count_nodes(psi, tol=1e-6):
    s = psi[np.abs(psi) > tol * np.abs(psi).max()]
    return int(np.sum(np.sign(s[1:]) != np.sign(s[:-1])))


L = np.pi
A_SEMI, V0 = np.pi, 12.0
LAM = 0.1
D_M, A_M = 12.0, 1.0

models = {
    "infinite": dict(title="Бесконечная яма",
                     V=lambda x: np.zeros_like(x), a=0.0, b=L, N=3000, k=6),
    "semi": dict(title="Полубесконечная яма",
                 V=lambda x: np.where(x < A_SEMI, 0.0, V0), a=0.0, b=A_SEMI + 30, N=12000, k=8),
    "harmonic": dict(title="Гармонический осциллятор",
                     V=lambda x: 0.5 * x**2, a=-10.0, b=10.0, N=4000, k=6),
    "quartic": dict(title="Ангармонический x^4",
                    V=lambda x: 0.5 * x**2 + LAM * x**4, a=-8.0, b=8.0, N=4000, k=6),
    "morse": dict(title="Морзе",
                  V=lambda x: D_M * (1 - np.exp(-A_M * x)) ** 2, a=-2.0, b=40.0, N=12000, k=8),
}


def exact_infinite(n):
    return n**2 * np.pi**2 / (2 * L**2)


def exact_harmonic(n):
    return n + 0.5


def exact_morse(n):
    w0 = A_M * np.sqrt(2 * D_M)
    return w0 * (n + 0.5) - (w0 * (n + 0.5)) ** 2 / (4 * D_M)


def morse_nmax():
    return int(np.floor(np.sqrt(2 * D_M) / A_M - 0.5))


def exact_semi():
    # корни k*ctg(ka) = -kappa, переписала без деления чтобы не было полюсов
    kmax = np.sqrt(2 * V0)
    f = lambda k: k * np.cos(k * A_SEMI) + np.sqrt(kmax**2 - k**2) * np.sin(k * A_SEMI)
    ks, grid = [], np.linspace(1e-6, kmax - 1e-9, 20000)
    vals = f(grid)
    for i in range(len(grid) - 1):
        if vals[i] * vals[i + 1] < 0:
            ks.append(brentq(f, grid[i], grid[i + 1]))
    return [k**2 / 2 for k in ks]


def quartic_first_order(n):
    # теория возмущений, <n|x^4|n> = 3(2n^2 + 2n + 1)/4
    return n + 0.5 + LAM * 0.75 * (2 * n**2 + 2 * n + 1)


def run():
    out = {"params": dict(L=L, A_SEMI=A_SEMI, V0=V0, LAM=LAM, D_M=D_M, A_M=A_M), "models": {}}

    for key, m in models.items():
        x, E, psi = solve(m["V"], m["a"], m["b"], m["N"], m["k"])
        h = x[1] - x[0]
        ortho_err = float(np.abs(psi.T @ psi * h - np.eye(m["k"])).max())
        out["models"][key] = dict(
            title=m["title"],
            E=E.tolist(),
            nodes=[count_nodes(psi[:, n]) for n in range(m["k"])],
            p_forbidden=[forbidden_probability(x, m["V"], E[n], psi[:, n]) for n in range(m["k"])],
            ortho_err=ortho_err,
            h=h,
        )

    out["models"]["infinite"]["exact"] = [exact_infinite(n) for n in range(1, 7)]
    out["models"]["harmonic"]["exact"] = [exact_harmonic(n) for n in range(6)]
    out["models"]["semi"]["exact"] = exact_semi()
    out["models"]["morse"]["exact"] = [exact_morse(n) for n in range(morse_nmax() + 1)]
    out["models"]["quartic"]["pt1"] = [quartic_first_order(n) for n in range(6)]

    # для x^4 точного нет, беру ричардсона: две сетки, ошибка h^2 -> h^4
    m = models["quartic"]
    _, E1, _ = solve(m["V"], m["a"], m["b"], 4000, 6)
    _, E2, _ = solve(m["V"], m["a"], m["b"], 8001, 6)  # 8001 чтобы шаг был ровно в 2 раза меньше
    out["models"]["quartic"]["richardson"] = ((4 * E2 - E1) / 3).tolist()

    out["continuum"] = continuum_check()
    out["convergence"] = convergence()
    out["discrete_exact"] = discrete_exact_check()
    out["hydrogen"] = hydrogen()
    out["checks"] = dict(harmonic_ground_forbidden_exact=float(erfc(1.0)),
                         hydrogen_1s_forbidden_exact=float(13 * np.exp(-4)))
    return out


def convergence():
    # наклон log(err) от log(h) должен быть 2
    res = {}
    Ns = [25, 50, 100, 200, 400, 800, 1600]
    for key, exact, n_off in [("infinite", exact_infinite, 1), ("harmonic", exact_harmonic, 0)]:
        m = models[key]
        rows = []
        for N in Ns:
            x, E, _ = solve(m["V"], m["a"], m["b"], N, 3)
            rows.append(dict(N=N, h=x[1] - x[0],
                             err=[abs(E[i] - exact(i + n_off)) for i in range(3)]))
        hs = np.log([r["h"] for r in rows])
        slopes = [float(np.polyfit(hs, np.log([r["err"][i] for r in rows]), 1)[0]) for i in range(3)]
        res[key] = dict(rows=rows, slopes=slopes)
    return res


def continuum_check():
    # уровни выше порога это просто стоячие волны в моей коробке:
    # если удлинить область, связанные не меняются, а эти съезжают к порогу
    res = {}
    for key, thr in [("semi", V0), ("morse", D_M)]:
        m = models[key]
        rows = []
        for extra in (1.0, 2.0):
            b = m["a"] + (m["b"] - m["a"]) * extra
            _, E, _ = solve(m["V"], m["a"], b, int(m["N"] * extra), m["k"])
            rows.append(dict(b=b, E=E.tolist()))
        res[key] = dict(threshold=thr, rows=rows)
    return res


def discrete_exact_check():
    # для ямы сеточная задача решается точно: E_n(h) = (2/h^2) * sin^2(n*pi*h / 2L)
    N = 200
    m = models["infinite"]
    x, E, _ = solve(m["V"], m["a"], m["b"], N, 6)
    h = x[1] - x[0]
    Eh = [(2 / h**2) * np.sin(n * np.pi * h / (2 * L)) ** 2 for n in range(1, 7)]
    return dict(N=N, h=h, numeric=E.tolist(), formula=Eh,
                max_diff=float(np.max(np.abs(np.array(Eh) - E))))


def hydrogen():
    # s-состояния: для u = r*R уравнение такое же одномерное, u(0) = 0
    V = lambda r: -1.0 / r
    r, E, u = solve(V, 0.0, 250.0, 25000, 6)
    exact = [-1 / (2 * n**2) for n in range(1, 7)]
    return dict(E=E.tolist(), exact=exact,
                p_forbidden_1s=forbidden_probability(r, V, E[0], u[:, 0]),
                p_forbidden=[forbidden_probability(r, V, E[n], u[:, n]) for n in range(6)])


if __name__ == "__main__":
    res = run()
    np.set_printoptions(precision=6, suppress=True)
    for key, m in res["models"].items():
        print(f"\n=== {m['title']}  (h = {m['h']:.4g})")
        print("  E числ.:", np.round(m["E"], 6))
        for name in ("exact", "pt1", "richardson"):
            if name in m:
                print(f"  {name:10s}:", np.round(m[name], 6))
        print("  узлов:", m["nodes"])
        print("  P_запр:", np.round(m["p_forbidden"], 4))
        print(f"  ортонорм. ошибка: {m['ortho_err']:.1e}")
    print("\n=== наклон сходимости:")
    for k, v in res["convergence"].items():
        print(" ", k, np.round(v["slopes"], 3))
    print(f"\n=== сетка против точной формулы: {res['discrete_exact']['max_diff']:.2e}")
    hy = res["hydrogen"]
    print("\n=== водород E:", np.round(hy["E"], 5), "\n   точно:", np.round(hy["exact"], 5))
    print("   P_запр(1s) =", round(hy["p_forbidden_1s"], 4), " теория 13e^-4 =", round(res["checks"]["hydrogen_1s_forbidden_exact"], 4))
    print("   P_запр(осц. n=0) теория erfc(1) =", round(res["checks"]["harmonic_ground_forbidden_exact"], 4))
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)