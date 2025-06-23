"""
ФПИК 09.03.01 Технологии экспериментальных исследований.

Контрольная работа 5-6. Сергеев Денис Олегович. ВТз-361с.
"""
import math
import sys
import textwrap
from dataclasses import dataclass
from typing import List, Tuple, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ───────────────────────────── ДАННЫЕ ДЛЯ 11 ВАРИАНТА ──────────────────────
RAW_DATA: List[int] = [
    15, 12, 16, 12, 13, 16, 10, 16, 14, 17,
    13, 15, 13, 14, 13, 14, 15, 17, 11, 15,
    16, 15, 12, 13, 11, 14, 11, 15, 13, 12,
    9,  13, 13, 16, 14, 11, 13, 14, 17, 15,
    13, 16, 13, 14, 13, 16, 16, 10, 14, 9,
    9,  13, 14, 13, 11, 13, 11, 13, 12, 13,
    14, 13, 13, 14, 15, 13, 13, 12, 13, 18,
    12, 10, 14, 15, 12, 11, 14, 14, 18, 15,
    14, 14, 15, 13, 15, 14, 9,  13, 18, 12,
    10, 14, 15, 13, 12, 16, 14, 10, 14, 16,
]

INTERVALS: List[Tuple[float, float, int]] = [
    (8, 10, 8),
    (10, 12, 24),
    (12, 14, 36),
    (14, 16, 23),
    (16, 18, 9),
]

# ───────────────────────── ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ─────────────────────────


def format_table(header: Sequence[str], rows_iter) -> str:
    """Формирует отформатированную таблицу в виде строки."""
    rows = [tuple(map(str, r))
            for r in rows_iter]
    header = [str(h) for h in header]
    n_cols = len(header)

    widths = []
    for i in range(n_cols):
        col_cells = [header[i]] + [row[i] for row in rows]
        widths.append(max(len(c) for c in col_cells))

    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    sep = "-+-".join("-" * w for w in widths)

    out_lines = [fmt.format(*header), sep]
    out_lines += [fmt.format(*row) for row in rows]
    return "\n".join(out_lines)

# ───────────────────────── СТАТИСТИЧЕСКИЙ АНАЛИЗ ─────────────────────────


def variation_series(data: Sequence[float]) -> np.ndarray:
    """Возвращает вариационный ряд (отсортированные данные)."""
    return np.sort(data)


def frequency_table(data: Sequence[float]):
    """Возвращает уникальные значения, абсолютные и относительные частоты."""
    vals, counts = np.unique(data, return_counts=True)
    n = len(data)
    rel = counts / n
    return vals, counts, rel


def empirical_cdf(data: Sequence[float]) -> Callable[[float], float]:
    """Строит эмпирическую функцию распределения по данным."""
    sorted_data = variation_series(data)
    n = len(sorted_data)

    def to_the_float(x: float) -> float:
        return float(np.searchsorted(sorted_data, x, side="right") / n)

    return to_the_float


def sample_statistics(data: Sequence[float]):
    """
    Вычисляет выборочные характеристики.

    Включает среднее значение, дисперсии (смещённую и несмещённую),
    а также стандартное отклонение.
    """
    x = np.asarray(data)
    n = len(x)
    mean = x.mean()
    var_b = x.var(ddof=0)
    var_u = x.var(ddof=1)
    std_u = math.sqrt(var_u)
    return {
        "n": n,
        "mean": mean,
        "var (biased)": var_b,
        "var (unbiased)": var_u,
        "std (unbiased)": std_u,
    }

# ───────────────────────── КРИТЕРИЙ ПИРСОНА ─────────────────────────


@dataclass
class ChiSqResult:
    """Структура для хранения результатов критерия Пирсона."""

    chi2: float
    df: int
    p: float
    reject: bool


def _merge_low_expected(
    obs: np.ndarray,
    exp: np.ndarray,
    threshold: float = 5
):
    """Объединяет интервалы с ожидаемыми частотами ниже порога."""
    tmp_o = tmp_e = 0.0
    o_m, e_m = [], []
    for o, e in zip(obs, exp):
        tmp_o += o
        tmp_e += e
        if tmp_e >= threshold:
            o_m.append(tmp_o)
            e_m.append(tmp_e)
            tmp_o = tmp_e = 0.0
    if tmp_e:
        o_m[-1] += tmp_o
        e_m[-1] += tmp_e
    return np.array(o_m), np.array(e_m)


def chi_square_normal_raw(
    data: Sequence[float],
    alpha: float = 0.05
) -> ChiSqResult:
    """
    Проверяет нормальность распределения сырых данных.

    Использует критерий согласия Пирсона (χ²) с автоматическим определением
    интервалов по правилу Стерджеса. Возвращает значение χ², число степеней
    свободы, p-значение и результат проверки гипотезы при значимости alpha.
    """
    data = np.asarray(data)
    n = data.size
    μ = data.mean()
    σ = data.std(ddof=1)

    k = int(np.floor(1 + 3.322 * math.log10(n)))
    probs = np.linspace(0, 1, k + 1)
    edges = stats.norm.ppf(probs, loc=μ, scale=σ)
    edges[0], edges[-1] = -np.inf, np.inf
    observed, _ = np.histogram(data, bins=edges)
    expected = n * np.diff(probs)

    observed, expected = _merge_low_expected(observed, expected)
    χ2 = ((observed - expected) ** 2 / expected).sum()
    df = len(observed) - 1 - 2
    p = 1 - stats.chi2.cdf(χ2, df)
    return ChiSqResult(χ2, df, p, p < alpha)


def chi_square_normal_interval(
    bins: List[Tuple[float, float, int]],
    alpha: float = 0.05
) -> ChiSqResult:
    """
    Проверяет нормальность распределения по интервальному ряду.

    Использует критерий Пирсона (χ²) для анализа частот в заданных интервалах.
    Интервалы описываются тройками (нижняя граница, верхняя граница, частота).
    Вычисляются ожидаемые значения, проводится объединение редких интервалов
    и возвращается результат проверки на заданном уровне значимости alpha.
    """
    counts = np.array([c for *_, c in bins])
    n = counts.sum()

    lows = np.array([a for a, *_ in bins], dtype=float)
    highs = np.array([b for _, b, *_ in bins], dtype=float)
    mids = (lows + highs) / 2

    μ = (mids * counts).sum() / n
    σ = math.sqrt(((mids - μ) ** 2 * counts).sum() / (n - 1))

    expected = n * (stats.norm.cdf(highs, μ, σ) - stats.norm.cdf(lows, μ, σ))
    observed, expected = _merge_low_expected(counts, expected)

    χ2 = ((observed - expected) ** 2 / expected).sum()
    df = len(observed) - 1 - 2
    p = 1 - stats.chi2.cdf(χ2, df)
    return ChiSqResult(χ2, df, p, p < alpha)

# ─────────────────────────────── ГРАФИКИ ───────────────────────────────────


_DEF_STYLE = {
    "linewidth": 1.8,
    "marker": "o",
    "markersize": 4,
}


def plot_polygon(vals: Sequence[float], counts: Sequence[int], title: str):
    """Рисует полигон частот по значениям и количествам."""
    plt.figure()
    plt.plot(vals, counts, **_DEF_STYLE)
    plt.xlabel("x")
    plt.ylabel("n_i")
    plt.title(title)
    plt.grid(True, linestyle=":")
    plt.show()


def plot_histogram(
    data: Sequence[float],
    bins: int | Sequence[float],
    title: str, density: bool = False
):
    """Строит гистограмму частот или плотности по данным."""
    plt.figure()
    plt.hist(data, bins=bins, density=density, edgecolor="black", alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("density" if density else "frequency")
    plt.title(title)
    plt.grid(True, linestyle=":")
    plt.show()


def plot_ecdf(data: Sequence[float], title: str):
    """Строит график эмпирической функции распределения."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    y = np.arange(1, n + 1) / n
    plt.figure()
    plt.step(sorted_data, y, where="post")
    plt.xlabel("x")
    plt.ylabel("to_the_float*(x)")
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":")
    plt.show()

# ──────────────────────────────── МЕНЮ ──────────────────────────────────


_MENU_TEXT = """
\n— Лабораторные работы №5 и №6 — вариант 11 — Сергеев Д.О. —
1  Вариационный ряд (сырые данные)
2  Статистический ряд (сырые данные)
3  Эмпирическая функция распределения (сырые данные)
4  Числовые характеристики (сырые данные)
5  Проверка нормальности (сырые данные)
6  Интервальный ряд (показать данные)
7  Проверка нормальности (интервальный ряд)
8  Построить графики (сырые данные)
9  Информация
0  Выход
"""


def menu():
    """Запускает консольное меню для выбора действия."""
    while True:
        print(_MENU_TEXT)
        choice = input("Ваш выбор: ").strip()
        if choice == "1":
            print("Вариационный ряд:")
            print(variation_series(RAW_DATA))
        elif choice == "2":
            vals, n_i, r_i = frequency_table(RAW_DATA)
            print(format_table(["x", "n_i", "r_i"], zip(
                vals, n_i, map(lambda r: f"{r:.3f}", r_i))))
        elif choice == "3":
            to_the_float = empirical_cdf(RAW_DATA)
            print(
                "to_the_float*(x) = k/n, где k — число наблюдений ≤ x. "
                "Например:"
            )
            for x in [10, 12, 14, 16, 18]:
                print(f"to_the_float*({x}) = {to_the_float(x):.3f}")
            plot_ecdf(
                RAW_DATA,
                "Эмпирическая функция распределения (raw data)"
            )
        elif choice == "4":
            for k, v in sample_statistics(RAW_DATA).items():
                print(f"{k:<17}: {v}")
        elif choice == "5":
            res = chi_square_normal_raw(RAW_DATA)
            print(
                f"χ² = {res.chi2:.4f}, df = {res.df}, p = {res.p:.4f} → ",
                end=""
            )
            print(
                "Гипотеза отвергается"
                if res.reject else "Нет оснований отвергнуть гипотезу"
            )
        elif choice == "6":
            print(format_table(["[a,b)", "n_i"],
                  ((f"[{a},{b})", c) for a, b, c in INTERVALS)))
        elif choice == "7":
            res = chi_square_normal_interval(INTERVALS)
            print(
                f"χ² = {res.chi2:.4f}, df = {res.df}, p = {res.p:.4f} → ",
                end=""
            )
            print(
                "Гипотеза отвергается"
                if res.reject else "Нет оснований отвергнуть гипотезу"
            )
        elif choice == "8":
            vals, n_i, _ = frequency_table(RAW_DATA)
            plot_polygon(vals, n_i, "Полигон частот (raw data)")
            plot_histogram(RAW_DATA, bins="sturges",
                           title="Гистограмма частот (raw data)")
            plot_histogram(RAW_DATA, bins="sturges",
                           title="Гистограмма относительных частот",
                           density=True
                           )
        elif choice == "9":
            print(
                "Обе работы были выполнены в одном коде для удобства "
                "демонстрации, а так же код был приведен к стандарту "
                "PEP8 для удобства чтения."
            )
        elif choice == "0":
            print("До свидания!")
            break
        else:
            print("Ошибка: выберите пункт из меню.")


if __name__ == "__main__":
    """
    Точка входа в программу при запуске как основного скрипта.
    Если стандартный ввод интерактивен, запускается меню.
    Иначе выводится предупреждение о невозможности взаимодействия.
    """
    if sys.stdin.isatty():
        menu()
    else:
        print(textwrap.dedent("""
            Запуск меню невозможен: стандартный ввод не интерактивен.
            Запустите скрипт в терминале или передайте управление
            функции menu() из своего кода.
        """))
