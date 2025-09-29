# sigman.py
from __future__ import annotations
from decimal import localcontext, ROUND_HALF_EVEN
import math
from typing import Iterable, List, Literal, Union, overload
from enum import Enum
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP, ROUND_HALF_DOWN, \
                     ROUND_DOWN, ROUND_FLOOR, \
                    ROUND_CEILING, ROUND_UP

NumberLike = Union[int, float, str, Decimal]

class RoundRule(Enum):
    BANKERS   = "BANKERS"    # ROUND_HALF_EVEN（四舍六入五凑偶）
    HALF_UP   = "HALF_UP"    # 传统四舍五入
    HALF_DOWN = "HALF_DOWN"  # 四舍五不入
    TRUNC     = "TRUNC"      # 截断（向 0）
    FLOOR     = "FLOOR"      # 向下
    CEIL      = "CEIL"       # 向上
    UP        = "UP"         # 远离 0


_ROUNDING_MAP = {
    RoundRule.BANKERS:   ROUND_HALF_EVEN,
    RoundRule.HALF_UP:   ROUND_HALF_UP,
    RoundRule.HALF_DOWN: ROUND_HALF_DOWN,
    RoundRule.TRUNC:     ROUND_DOWN,
    RoundRule.FLOOR:     ROUND_FLOOR,
    RoundRule.CEIL:      ROUND_CEILING,
    RoundRule.UP:        ROUND_UP,
}


def _to_decimal(x: NumberLike) -> Decimal:
    """
    将输入稳健地转为 Decimal：
    - float 通过 str() 以减少二进制浮点误差影响
    - str/Decimal 直接转
    - int 直接转
    """
    if isinstance(x, Decimal):
        return x
    if isinstance(x, float):
        return Decimal(str(x))
    return Decimal(str(x))

def _quant_exp_for_sig(x: Decimal, sig: int) -> int:
    """
    计算按“有效数字 sig 位”量化所需的 10 的指数 e，
    使得 quant = Decimal('1e' + str(e))，对 x 进行 quantize。
    """
    if x.is_zero():
        # 对 0，保留 sig 位有效数字 → 小数位数 = sig - 1
        return -(sig - 1)
    # x = m * 10^k, 其中 k = floor(log10(|x|))
    k = int(math.floor(math.log10(abs(x))))
    # 目标是在第 (k - sig + 1) 位进行量化
    return k - sig + 1

def _quant_exp_for_dec_places(places: int) -> int:
    # places=2 → 量化到 1e-2
    return -places

class SignificantDigitsManager:
    """
    有效数字管理系统

    功能：
    - round_sig:  按“有效数字位数”舍入（四舍六入五凑偶）
    - round_dec:  按“小数位数”舍入（四舍六入五凑偶）
    - format_sig: 按“有效数字位数”返回字符串（可补零、可科学计数法）
    - format_dec: 按“小数位数”返回字符串（补零）
    - batch_*:    批处理列表/可迭代
    """

    def __init__(
        self,
        default_rule: RoundRule = RoundRule.BANKERS,
        # 计算时的上下文精度冗余（避免中间步骤精度不足）
        extra_precision: int = 10,
        sci_threshold_low: int = -6,
        sci_threshold_high: int = 6,
    ):
        """
        参数:
            default_rule: 默认舍入规则（本系统实现“四舍六入五凑偶”）
            extra_precision: 计算临时精度冗余位数
            sci_threshold_low / sci_threshold_high:
                format_sig 自动选择是否使用科学计数法的阈值:
                若指数 k < sci_threshold_low 或 k >= sci_threshold_high，则用科学计数法
        """
        self.default_rule = default_rule
        self.extra_precision = extra_precision
        self.sci_threshold_low = sci_threshold_low
        self.sci_threshold_high = sci_threshold_high

    # -------------------- 核心数值舍入 --------------------
    def round_sig(self, x: NumberLike, sig: int, rule: RoundRule | None = None) -> Decimal:
        """
        按“有效数字 sig 位”舍入，返回 Decimal。
        采用“四舍六入五凑偶”（银行家舍入）。
        """
        if sig <= 0:
            raise ValueError("sig 必须为正整数。")
        rule = rule or self.default_rule
        rounding = _ROUNDING_MAP[rule]

        dx = _to_decimal(x)
        if dx.is_nan() or dx.is_infinite():
            return dx  # 保持 NaN/Inf

        e = _quant_exp_for_sig(dx, sig)
        quant = Decimal(f"1e{e}")

        # 设置临时高精度上下文，避免中间精度不足
        with localcontext() as ctx:
            ctx.prec = max(sig + self.extra_precision, 28)
            ctx.rounding = rounding
            return dx.quantize(quant)  # 银行家舍入由 ctx.rounding 控制

    def round_dec(self, x: NumberLike, places: int, rule: RoundRule | None = None) -> Decimal:
        """
        按“小数位 places”舍入，返回 Decimal。
        采用“四舍六入五凑偶”（银行家舍入）。
        """
        if places < 0:
            raise ValueError("places 不能为负。")
        rule = rule or self.default_rule
        rounding = _ROUNDING_MAP[rule]

        dx = _to_decimal(x)
        if dx.is_nan() or dx.is_infinite():
            return dx

        e = _quant_exp_for_dec_places(places)
        quant = Decimal(f"1e{e}")

        with localcontext() as ctx:
            ctx.prec = max(places + self.extra_precision, 28)
            ctx.rounding = rounding
            return dx.quantize(quant)

    # -------------------- 格式化输出（补零/科学计数法） --------------------
    def format_sig(
        self,
        x: NumberLike,
        sig: int,
        rule: RoundRule | None = None,
        scientific: bool | None = None,
    ) -> str:
        """
        按“有效数字 sig 位”舍入并格式化为字符串。
        - 自动或强制科学计数法
        - 补齐末尾 0，严格显示 sig 位有效数字

        参数:
            scientific:
                None → 根据阈值自动决定
                True → 强制科学计数法
                False → 尽量用普通十进制（可能包含很多小数位）
        """
        d = self.round_sig(x, sig, rule)

        # 0 的特殊处理：0.00...(sig-1 个 0)
        if d.is_zero():
            # 0.xxx 共 sig 位有效数字，即小数位 = sig-1
            frac = "0" * (sig - 1)
            return f"0.{frac}" if sig > 1 else "0"

        k = int(math.floor(math.log10(abs(d))))
        use_sci = scientific if scientific is not None else (k < self.sci_threshold_low or k >= self.sci_threshold_high)

        if use_sci:
            # 科学计数法：先把数转为 1.xxx * 10^k，再确保小数位数 = sig-1
            m = d.scaleb(-k)  # d = m * 10^k, 使得 m ∈ [1,10)
            # 把 m 量化到 sig-1 位小数，保证总有效数字 sig
            with localcontext() as ctx:
                ctx.prec = max(sig + self.extra_precision, 28)
                ctx.rounding = _ROUNDING_MAP[self.default_rule]
                m_q = m.quantize(Decimal(f"1e-{sig-1}"))
            s = format(m_q, "f")  # 十进制字符串
            # 补零（quantize 已经保证位数，一般不需要，但保险处理）
            if "." in s:
                int_part, frac_part = s.split(".")
                frac_part = frac_part.ljust(sig - 1, "0")
                s = f"{int_part}.{frac_part}" if sig > 1 else int_part
            else:
                if sig > 1:
                    s = s + "." + ("0" * (sig - 1))
            return f"{s}e{k:+d}"
        else:
            # 普通十进制：需要确保显示的有效数字位数为 sig（含补零）
            # 思路：转成字符串后，如无足够小数位则补 0；若位数过多也不要多显示
            # 方案：计算应有的小数位数 = max(sig - (k+1), 0)
            places = max(sig - (k + 1), 0)
            with localcontext() as ctx:
                ctx.prec = max(sig + self.extra_precision, 28)
                ctx.rounding = _ROUNDING_MAP[self.default_rule]
                dq = d.quantize(Decimal(f"1e-{places}"))
            s = format(dq, "f")
            # 补零至 places 位
            if places > 0:
                if "." in s:
                    int_part, frac_part = s.split(".")
                    s = int_part + "." + frac_part.ljust(places, "0")
                else:
                    s = s + "." + ("0" * places)
            return s

    def format_dec(
        self,
        x: NumberLike,
        places: int,
        rule: RoundRule | None = None,
    ) -> str:
        """
        按“小数位 places”舍入并格式化为字符串，强制补齐 places 位。
        """
        d = self.round_dec(x, places, rule)
        s = format(d, "f")
        if places == 0:
            # 去掉可能的尾随小数点与 0
            return s.split(".")[0]
        if "." in s:
            int_part, frac_part = s.split(".")
            return int_part + "." + frac_part.ljust(places, "0")
        else:
            return s + "." + ("0" * places)

    def infer_dec_places(self,s: str) -> int:
        """推断小数位数（用于加减规则）。"""
        s = s.strip().lower()
        if 'e' in s:
            base, expo = s.split('e')
            expo = int(expo)
            if '.' in base:
                frac = len(base.split('.')[1])
                return max(frac - expo, 0)
            else:
                return max(0 - expo, 0)
        else:
            return len(s.split('.')[1]) if '.' in s else 0

    def infer_sig_digits(self,s: str) -> int:
        """推断字面量的有效数字位数（用于乘除规则）。"""
        s = s.strip().lower()
        if 'e' in s:
            base, _ = s.split('e')
            # 科学计数法中：有效数字 = 去掉小数点后的总位数，去掉所有前导0
            digits = [c for c in base if c.isdigit()]
            # 去掉前导零
            while digits and digits[0] == '0':
                digits.pop(0)
            return max(len(digits), 1)
        else:
            if '.' in s:
                # 有小数点：所有非零及中间的0都算；末尾0也算（因为小数点显式给出）
                digits = [c for c in s if c.isdigit()]
                # 去掉整体前导零（例如 0.00340）
                i = 0
                while i < len(digits) and digits[i] == '0':
                    i += 1
                return max(len(digits) - i, 1)
            else:
                # 纯整数且无小数点：末尾0通常不算有效数字（标准教材约定）
                # 例如 1200 → 2 位有效数字（“1”“2”），末尾两个0不计
                # 若要让末尾0生效，请用科学计数法写成 1.200e3
                s_ = s.lstrip('+-')
                # 去掉末尾0
                s_ = s_.rstrip('0')
                # 去掉前导0
                s_ = s_.lstrip('0')
                return len(s_) if s_ else 1


"""
api参数表
#使用默认参数
mgr = SignificantDigitsManager()
#使用制定规则的数据处理系统
num_five1 = mgr.round_sig(25856,3)
num_five2 = mgr.round_dec(2659.25896,3)
#使用更加精确的字符串表示法，可以补零，可以科学计数
num_str1 = mgr.format_sig(26524.61,5)
num_str2 = mgr.format_dec(265.23,5)
num_str3 = mgr.format_sig(2623,5, scientific=True)
#检测一个使用字符串表示的有效数字的位数
places = mgr.infer_dec_places(num_str1)
sigs = mgr.infer_sig_digits(num_str1)
print(num_str1)
"""





