# -*- coding: utf-8 -*-
"""
analysis/line_fit_with_decimals.py
在你的 decimal_calculator / decimal_manager 体系上实现：
- 通过 DataProcessor.rect 切块（返回“字符串表”）
- 逐差法（k, b, k_i 的样本标准差）
- 最小二乘直线 y = a x + b（a,b 及其标准误差、R^2）
- 所有内部数值用 Decimal 计算；最终展示用你的 mgr/calc 规则格式化
"""

from __future__ import annotations
from typing import Iterable, Tuple, Dict, Any, Optional, List
from decimal import Decimal, getcontext

import pandas as pd

# —— 你的系统：有效数字与计算器 ——
from my_decimal.decimal_manager import SignificantDigitsManager, _to_decimal, RoundRule
from my_decimal.decimal_calculator import _place_exp_from_sigma, _round_y_to_place
from my_decimal.decimal_calculator import DecimalCalculator
from error.error_manager import DataProcessor
import parameter

mgr  = SignificantDigitsManager()
calc = DecimalCalculator()
pro  = DataProcessor()

getcontext().prec = 80  # 充足精度，避免统计运算中间误差

# =============== 小工具：把块转 x,y（Decimal 列表） ===============
def _to_xy_decimal(block: pd.DataFrame) -> tuple[List[Decimal], List[Decimal]]:
    """块前两列为 x,y；元素是字符串，转 Decimal；非法空串跳过。"""
    if block.shape[1] < 2:
        raise ValueError("切出的块列数不足 2 列（需要 x 与 y）。")

    blk = block.iloc[:, :2].copy()
    xs: List[Decimal] = []
    ys: List[Decimal] = []
    for r in blk.index:
        x_str = str(blk.iat[blk.index.get_loc(r), 0]).strip()
        y_str = str(blk.iat[blk.index.get_loc(r), 1]).strip()
        if x_str == "" or y_str == "":
            continue
        try:
            xs.append(_to_decimal(x_str))
            ys.append(_to_decimal(y_str))
        except Exception:
            # 非法数字行直接丢弃
            continue
    if len(xs) < 2:
        raise ValueError("有效数据不足（<2 行）。")
    return xs, ys

# =============== 逐差法（Decimal 版本） ===============
def successive_differences_decimal(xs: List[Decimal], ys: List[Decimal], m: int = 1):
    """
    逐差法：k_i = (y_{i+m}-y_i)/(x_{i+m}-x_i)；k=mean(k_i)，b = ȳ - k x̄
    非等间距 x 也严格使用 Δy/Δx（更稳健）。
    """
    n = len(xs)
    if n <= m:
        raise ValueError(f"数据量 {n} 过少，无法使用 m={m} 的逐差法。")

    ks: List[Decimal] = []
    for i in range(n - m):
        dx = xs[i + m] - xs[i]
        if dx == 0:
            continue
        dy = ys[i + m] - ys[i]
        ks.append(dy / dx)
    if not ks:
        raise ValueError("Δx 出现 0，无法计算逐差斜率。")

    nK = Decimal(len(ks))
    k  = sum(ks, Decimal(0)) / nK
    xbar = sum(xs, Decimal(0)) / Decimal(n)
    ybar = sum(ys, Decimal(0)) / Decimal(n)
    b  = ybar - k * xbar

    if len(ks) > 1:
        mu = k
        s2 = sum((ki - mu) * (ki - mu) for ki in ks) / Decimal(len(ks) - 1)
        k_std = s2.sqrt()
    else:
        k_std = Decimal(0)

    # 等间距提示
    dxs = [xs[i + 1] - xs[i] for i in range(n - 1)]
    uniform = all(d == dxs[0] for d in dxs)

    return {
        "k": k, "b": b, "k_list": ks, "k_std": k_std, "m": m,
        "note": "x 等间距" if uniform else "x 非等间距（按各自 Δx 处理）"
    }

# =============== 最小二乘直线（Decimal 版本） ===============
def least_squares_decimal(xs: List[Decimal], ys: List[Decimal]):
    """
    OLS: y = a x + b；返回 a,b, se_a, se_b, R^2。
    """
    n = len(xs)
    if n < 2:
        raise ValueError("最小二乘需要至少 2 个点。")

    xbar = sum(xs, Decimal(0)) / Decimal(n)
    ybar = sum(ys, Decimal(0)) / Decimal(n)

    Sxx = sum((x - xbar) * (x - xbar) for x in xs)
    Sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    if Sxx == 0:
        raise ValueError("Sxx=0（所有 x 相同），无法拟合直线。")

    a = Sxy / Sxx
    b = ybar - a * xbar

    resid = [y - (a * x + b) for x, y in zip(xs, ys)]
    if n > 2:
        s2 = sum(r * r for r in resid) / Decimal(n - 2)
        se_a = (s2 / Sxx).sqrt()
        se_b = (s2 * (Decimal(1) / Decimal(n) + xbar * xbar / Sxx)).sqrt()
    else:
        se_a = Decimal(0)
        se_b = Decimal(0)

    Syy = sum((y - ybar) * (y - ybar) for y in ys)
    r2 = Decimal(1) - (sum(r * r for r in resid) / Syy) if Syy != 0 else Decimal(1)

    return {"a": a, "b": b, "se_a": se_a, "se_b": se_b, "r2": r2, "n": n}

# =============== “值 ± σ” 的位权对齐显示（用你的内部工具） ===============
def format_with_sigma(value: Decimal, sigma: Decimal) -> tuple[str, str]:
    """
    以 σ 的位权对齐 value 与 σ；返回 (value_fmt, sigma_fmt)。
    """
    if sigma.is_zero():
        # σ=0 时降级：保留 1 位小数，仅示意
        return mgr.format_dec(value, 1), mgr.format_dec(Decimal(0), 1)
    p_place = _place_exp_from_sigma(sigma)   # 位权 10^p
    v_fmt   = _round_y_to_place(value, p_place)
    s_fmt   = _round_y_to_place(sigma, p_place)
    return v_fmt, s_fmt

# =============== 主入口：切块 ➜ 两种方法 ===============
def analyze_blocks_with_decimals(
    df: pd.DataFrame,
    rect_specs: Iterable[Tuple[Tuple[int, int], Tuple[int, int]]],
    *,
    diff_m: int = 1,
    fmt_sig_digits: int = 3,     # 仅用于单值展示（如 R^2 等无 σ 的量）
    by: str = "position",
) -> Dict[str, Dict[str, Any]]:
    """
    rect_specs: [((r1,r2),(c1,c2)), ...]  —— 1-based/闭区间 的“人类直觉坐标”
    实际调用你的 DataProcessor.rect，并用 parameter.dec_all 将块格式化为“字符串表”。
    返回字典：每块包含逐差与最小二乘的“Decimal 数值 + 已按你的规则格式化的字符串”。
    """
    results: Dict[str, Dict[str, Any]] = {}

    for idx, (rows, cols) in enumerate(rect_specs, start=1):
        # ① 切块（直接用你的 rect，并返回“字符串表”）
        blk = pro.rect(
            df,
            rows=rows, cols=cols,
            by=by, closed="both",
            dec_places=parameter.dec_all,  # 与你工程一致：统一为字符串表
        )

        # ② 转 Decimal & 计算
        xs, ys = _to_xy_decimal(blk)

        diff = successive_differences_decimal(xs, ys, m=diff_m)
        ols  = least_squares_decimal(xs, ys)

        # ③ 逐差法输出（k, b 与 k_std 做“值±σ”位权对齐）
        k_v, k_s = diff["k"], diff["k_std"]
        b_v      = diff["b"]
        k_fmt, k_sigma_fmt = format_with_sigma(k_v, k_s)
        # b 没有对应 σ，就用有效数字位数展示
        b_fmt = mgr.format_sig(b_v, fmt_sig_digits)

        # ④ 最小二乘输出（a,b 与各自标准误差位权对齐）
        a_v, b_v = ols["a"], ols["b"]
        se_a, se_b = ols["se_a"], ols["se_b"]
        a_fmt, se_a_fmt = format_with_sigma(a_v, se_a)
        b2_fmt, se_b_fmt = format_with_sigma(b_v, se_b)

        r2_fmt = mgr.format_sig(ols["r2"], min(fmt_sig_digits, 4))

        results[f"block#{idx}"] = {
            "rows": rows, "cols": cols, "n_points": len(xs),

            # 逐差法
            "diff": {
                "k": k_v, "k_std": k_s, "b": b_v, "m": diff_m, "note": diff["note"],
                "k_fmt": k_fmt, "k_std_fmt": k_sigma_fmt, "b_fmt": b_fmt,
            },

            # 最小二乘
            "ols": {
                "a": a_v, "b": b_v, "se_a": se_a, "se_b": se_b, "r2": ols["r2"], "n": ols["n"],
                "a_fmt": a_fmt, "se_a_fmt": se_a_fmt,
                "b_fmt": b2_fmt, "se_b_fmt": se_b_fmt,
                "r2_fmt": r2_fmt,
            },

            # 原始块（字符串表），方便你后续落表
            "block": blk,
        }

    return results
