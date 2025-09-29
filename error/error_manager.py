import pandas as pd
import copy as cp

import parameter
from my_decimal.decimal_manager import SignificantDigitsManager,_to_decimal
from my_decimal.decimal_calculator import DecimalCalculator, DecimalFormular
from my_decimal.decimal_calculator import _place_exp_from_sigma, _round_y_to_place
from file.file_manager import ExcelReader
import copy
import numpy as np
from typing import Iterable, Tuple, Any, Literal, Optional
from decimal import Decimal
import math as _m
mgr = SignificantDigitsManager()
calc = DecimalCalculator()
form = DecimalFormular()
exc = ExcelReader()

Closed = Literal["both", "left", "right", "neither"]
By     = Literal["label", "position"]

_ALLOWED_FUNCS = {
    "sin": _m.sin, "cos": _m.cos, "tan": _m.tan,
    "asin": _m.asin, "acos": _m.acos, "atan": _m.atan,
    "sinh": _m.sinh, "cosh": _m.cosh, "tanh": _m.tanh,
    "exp": _m.exp, "sqrt": _m.sqrt, "abs": abs,
    "ln": _m.log, "log": _m.log, "log10": _m.log10,
    "pi": _m.pi, "e": _m.e,
}

def _interval_mask_1d(idx: pd.Index, start: Any, end: Any, closed: Closed) -> pd.Series:
    left_ok  = {"both": idx >= start, "left": idx >= start, "right": idx > start,  "neither": idx > start}
    right_ok = {"both": idx <= end,   "left": idx <  end,   "right": idx <= end,   "neither": idx <  end}
    m_left  = left_ok[closed]  if start is not None else pd.Series(True, index=idx)
    m_right = right_ok[closed] if end   is not None else pd.Series(True, index=idx)
    return m_left & m_right

class DataProcessor:

    def rect(self,
            df: pd.DataFrame,
            rows: Tuple[Any, Any],
            cols: Tuple[Any, Any],
            *,
            by: By = "label",
            closed: Closed = "both",
            # ——字符串表输出控制——
            dec_places: Optional[int] = None,  # 按“小数位数”格式化；与 sig_digits 互斥
            sig_digits: Optional[int] = None,  # 按“有效数字位数”格式化；与 dec_places 互斥
            na_str: str = "",  # 缺失值在字符串表中的表示
    ) -> pd.DataFrame:
        """
        只返回“字符串表”的矩形块选择器。
        - by='label'  用 df.index / df.columns 的标签区间
        - by='position' 用 0 基位置区间（已处理左右端点闭开）
        - 数值 -> 按 dec_places 或 sig_digits 格式化成字符串
        - 其它类型 -> str(v)
        """
        # 1) 先按区间取出块（还是原有逻辑）
        if by == "label":
            rmask = _interval_mask_1d(df.index, rows[0], rows[1], closed)
            cmask = _interval_mask_1d(df.columns, cols[0], cols[1], closed)
            out = df.loc[rmask, cmask]
        else:
            nrow, ncol = len(df.index), len(df.columns)
            r0, r1 = rows;
            c0, c1 = cols
            r0 = 0 if r0 is None else int(r0)
            c0 = 0 if c0 is None else int(c0)
            r1 = nrow - 1 if r1 is None else int(r1)
            c1 = ncol - 1 if c1 is None else int(c1)
            incl_right = closed in ("both", "right")
            incl_left = closed in ("both", "left")
            r_start = r0 + (0 if incl_left else 1)
            c_start = c0 + (0 if incl_left else 1)
            r_stop = (r1 + 1) if incl_right else r1
            c_stop = (c1 + 1) if incl_right else c1
            r_start = max(0, r_start)
            c_start = max(0, c_start)
            r_stop = min(nrow, r_stop)
            c_stop = min(ncol, c_stop)
            if r_start >= r_stop or c_start >= c_stop:
                out = df.iloc[0:0, 0:0]
            else:
                out = df.iloc[r_start:r_stop, c_start:c_stop]

        # 2) 构造“字符串表”（不再返回数值表）
        sdf = out.copy().astype(object)
        for r in sdf.index:
            for c in sdf.columns:
                v = out.at[r, c]
                if pd.isna(v):
                    sdf.at[r, c] = na_str
                    continue
                # 数值：按指定规则格式化；其他：str
                if isinstance(v, (int, float, np.number, Decimal)):
                    if dec_places is not None and sig_digits is None:
                        sdf.at[r, c] = mgr.format_dec(float(v), dec_places)  # :contentReference[oaicite:2]{index=2}
                    elif sig_digits is not None and dec_places is None:
                        sdf.at[r, c] = mgr.format_sig(float(v), sig_digits)  # :contentReference[oaicite:3]{index=3}
                    else:
                        # 两个都没给或都给了：直接 str，不擅自决定规则
                        sdf.at[r, c] = str(v)
                else:
                    sdf.at[r, c] = str(v)
        return sdf
    def format_sig_batch(self, df: pd.DataFrame, dec:int, overwrite:bool = False) -> pd.DataFrame:
        if not overwrite:
            new_df = cp.deepcopy(df)
        else:
            new_df = df
        for r in df.index:
            for a in df.columns:
                new_df.at[r, a] = mgr.format_sig(df.at[r, a], dec)
        return new_df
    def format_dec_batch(self, df: pd.DataFrame, dec:int, overwrite:bool = False) -> pd.DataFrame:
        if not overwrite:
            new_df = cp.deepcopy(df)
        else:
            new_df = df
        for r in df.index:
            for a in df.columns:
                new_df.at[r, a] = mgr.format_dec(df.at[r, a], dec)
        return new_df
    def average(self, df: pd.DataFrame) -> str:
        aver_sum = "0"
        for r in df.index:
            for a in df.columns:
                aver_sum = calc.common_sum(df.at[r, a],aver_sum, parameter.dec_all)
        num = mgr.format_dec(df.size, 5)
        aver = calc.div_sig(aver_sum, num)
        return aver

    def standard_deviation(self, df: pd.DataFrame, average: str) -> str:
        ss = "0"  # sum of (xi - mean)^2
        n = df.size
        for r in df.index:
            for a in df.columns:
                v = df.at[r, a]
                diff = calc.sub_sig(v, average)  # xi - mean
                sq = calc.mul_sig(diff, diff)  # (xi - mean)^2
                ss = calc.sum_sig(ss, sq)
        expr = f"sqrt(x / ({n} * ({n} - 1)))"
        return form.eval_sig_expr(expr, ss)[0]
    def combined_uncertainty(self, standard_deviation: str, device_deviation: float, device_sig: int) -> list:
        device = copy.deepcopy(device_deviation)
        device = mgr.format_sig(device,device_sig)
        device = form.eval_sig_expr("x/sqrt(3)",device)
        device = device[0]
        expr = f"sqrt(({device} * {device}) + (x * x))"
        combined_uncertainty = form.eval_sig_expr(expr,standard_deviation)[0]
        return [combined_uncertainty, device]
    def relative_uncertainty(self, combined_deviation, average) -> str:
        ex = calc.div_sig(combined_deviation,average)
        return ex

    def _safe_eval(self, expr: str, varmap: dict[str, float]) -> float:
        """受限 eval：只开放上面的数学白名单和传入的变量。支持把 ^ 写成幂。"""
        code = compile(expr.replace("^", "**"), "<expr>", "eval")
        return float(eval(code, {"__builtins__": {}}, {**_ALLOWED_FUNCS, **varmap}))

    def indirect_uncertainty(self,
            expr: str,
            variables: dict[str, tuple[str, str]],
            *,
            # 步长选择：'place' 用 x_i 最后一位的位权（与现有函数一致）；'sigma' 用 σ_i 本身
            step: str = "place",
            # 相对不确定度显示的有效数字位数
            rel_sig_digits: int = 3,
    ):
        """
        计算间接不确定度和相对间接不确定度（支持多变量）。
        参数：
            expr: 形如 "x1*x2 + sin(x3)" 的函数表达式（变量名需与 variables 的键一致）
            variables: { 变量名: (x_i 的字面量字符串, σ_i 的字面量/数字) }
                       例：{"x1": ("12.30", "0.05"), "x2": ("3.00e2", "0.6")}
            step: 'place' 或 'sigma'，有限差分的步长选择
            rel_sig_digits: 相对不确定度显示用的有效数字位数
        返回：
            dict，包含数值与格式化结果：
            {
              "Y": <Decimal数值>, "sigma_y": <Decimal数值>, "E_y": <Decimal数值>,
              "Y_fmt": <与 sigma 对齐的字符串>, "sigma_fmt": <与 Y 同位权的字符串>,
              "E_fmt": <相对不确定度格式化字符串>
            }
        """
        # 1) 组装数值表
        xs_f = {name: float(_to_decimal(x_literal)) for name, (x_literal, _) in variables.items()}

        # 2) 计算 Y
        Y = self._safe_eval(expr, xs_f)
        Y_dec = _to_decimal(str(Y))

        # 3) 中心差分求各偏导 * σ_i
        terms = []
        for name, (x_lit, sigma_lit) in variables.items():
            x0 = xs_f[name]
            sigma_i = float(_to_decimal(sigma_lit))

            if step == "sigma" and sigma_i > 0:
                h = sigma_i
            else:
                # 用“最后一位位权”作步长（与 DecimalFormular 的做法一致）
                # 该位权 = 10^p_place, 通过对 σ_i 的 Decimal 直接取得数量级
                # 若 σ_i == 0，则退化到一个很小的步长
                sig_dec = _to_decimal(sigma_lit)
                if sig_dec.is_zero():
                    h = 1e-12 if x0 == 0 else abs(x0) * 1e-12
                else:
                    p_place = _place_exp_from_sigma(sig_dec)
                    h = 10.0 ** p_place

            # 只扰动当前变量
            xs_f[name] = x0 + h
            f_plus = self._safe_eval(expr, xs_f)
            xs_f[name] = x0 - h
            f_minus = self._safe_eval(expr, xs_f)
            xs_f[name] = x0  # 还原

            dfdxi = (f_plus - f_minus) / (2.0 * h)
            terms.append((dfdxi * sigma_i) ** 2)

        sigma_y = _m.sqrt(sum(terms))
        sigma_dec = _to_decimal(str(sigma_y))

        # 4) 相对不确定度
        E_y = (sigma_y / abs(Y)) if Y != 0 else float("inf")
        E_dec = _to_decimal(str(E_y)) if _m.isfinite(E_y) else _to_decimal("Infinity")

        # 5) 与 σ 对齐的显示（同你现有的格式化规则）
        if sigma_dec.is_zero():
            # 极端情形：σ=0，保留 1 位小数示意
            Y_fmt = mgr.format_dec(Y_dec, 1)
            sigma_fmt = mgr.format_dec(Decimal(0), 1)
        else:
            p_place = _place_exp_from_sigma(sigma_dec)
            Y_fmt = _round_y_to_place(Y_dec, p_place)
            sigma_fmt = _round_y_to_place(sigma_dec, p_place)

        E_fmt = mgr.format_sig(E_dec, rel_sig_digits)

        return {
            "Y": Y_dec,
            "sigma_y": sigma_dec,
            "E_y": E_dec,
            "Y_fmt": Y_fmt,
            "sigma_fmt": sigma_fmt,
            "E_fmt": E_fmt,
        }

# 函数：Y = x1 * x2 + sin(x3)

"""
df = pd.DataFrame(
    np.arange(1, 17).reshape(4, 4),
    index=[10, 20, 30, 40],
    columns=list("ABCD"),
)
pro = DataProcessor()
expr = "x1 * x2 + sin(x3)"
# 直接量与其标准不确定度（字面量字符串即可，与工程中其它接口一致）
vars_in = {
    "x1": ("12.30", "0.05"),     # x1 = 12.30,  σ1 = 0.05
    "x2": ("3.00e2", "0.6"),     # x2 = 3.00×10^2, σ2 = 0.6
    "x3": ("1.5708", "0.0003"),  # x3 = 1.5708, σ3 = 3e-4
}
out = pro.indirect_uncertainty(expr, vars_in, step="place", rel_sig_digits=3)
print("Y =", out["Y_fmt"])
print("σ_y =", out["sigma_fmt"])
print("E_y =", out["E_fmt"])
# 取行 [20, 40]、列 ['B', 'D'] 的矩形
blk = pro.rect(df, rows=(20, 40), cols=('B', 'D'), by="label", closed="both")
print(blk)
# 端点改为左闭右开：
blk2 = pro.rect(df, rows=(20, 40), cols=('B', 'D'), by="label", closed="left")
# 行区间(1,3)，列区间(0,2)，左闭右闭 => 包含行 1~3、列 0~2（注意 iloc 的右端已处理）
blk = pro.rect(df, rows=(1, 3), cols=(0, 2), by="position", closed="both")
# 左开右开：
blk_open = pro.rect(df, rows=(1, 3), cols=(0, 2), by="position", closed="neither")
# 无穷端点（None）：从开头到列 index=2（右闭）
blk_head = pro.rect(df, rows=(None, 2), cols=(None, 2), by="position", closed="right")
"""





