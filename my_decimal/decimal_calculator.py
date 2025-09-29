from decimal import Decimal
from .decimal_manager import SignificantDigitsManager,_to_decimal
import math
import ast

mgr = SignificantDigitsManager()

"""以下的部分均用于计算位权"""
def _sigma_exp_from_literal(s: str) -> int:
    """
    返回 p，使得 σx = 10**p。
    规则：
      - '7.10e3' -> base 有 2 位小数，指数 3 => p = 3 - 2 = +1(→ 百分位 10^1=10，但“最后一位”是百位，对应 1e2；
         注意：这里 σ 是“位权”，即 10^(e - 小数位数)。)
      - '12.30'  -> p = -2 (σ=0.01)
      - '1200'   -> p = 0   (无小数点，默认个位；若想让末尾 0 生效请用 1.200e3)
    """
    s = s.strip().lower()
    if 'e' in s:
        base, expo = s.split('e', 1)
        e = int(expo)
        frac = len(base.split('.')[1]) if '.' in base else 0
        return e - frac
    else:
        if '.' in s:
            return -len(s.split('.')[1])
        else:
            return 0  # 纯整数，默认最后一位在个位

def _place_exp_from_sigma(sigma: Decimal) -> int:
    """σy 的数量级 p = floor(log10(σy))，返回 p 使得“位权”=10^p。"""
    if sigma.is_zero():
        return 0
    # Decimal 自带 adjusted() 但对 <1 的数要小心；这里直接用对数
    return sigma.adjusted()
def _exp10(p: int) -> Decimal:
    return Decimal(10) ** p
def _round_y_to_place(y: Decimal, p_place: int) -> str:
    """
    把 y 的最后一位对齐到“位权=10^p_place”的那一位。
    做法：计算 y 的数量级 k（科学计数法的指数），
    需要的有效位数 sig = k - p_place + 1，然后用 mgr.format_sig。
    """
    if y.is_zero():
        # 0 的话：若 p_place<0，就显示相应小数位；否则就是 "0"
        if p_place < 0:
            return mgr.format_dec(Decimal(0), -p_place)
        return "0"
    k = y.adjusted()  # floor(log10(|y|))
    sig = k - p_place + 1
    sig = max(sig, 1)
    return mgr.format_sig(y, sig)


class DecimalCalculator:
    def sum_sig(self,a: str, b: str) -> str:

        val = _to_decimal(a) + _to_decimal(b)
        places = min(mgr.infer_dec_places(a), mgr.infer_dec_places(b))
        return mgr.format_dec(val, places)

    def sub_sig(self,a: str, b: str) -> str:
        vals = _to_decimal(a) - _to_decimal(b)
        places = min(mgr.infer_dec_places(a), mgr.infer_dec_places(b))
        return mgr.format_dec(vals, places)

    def mul_sig(self,a: str, b: str) -> str:
        """
        乘法：结果按“参与者中有效数字最少”的位数保留（银行家舍入）。
        """
        val = _to_decimal(a) * _to_decimal(b)
        sig = min(mgr.infer_sig_digits(a), mgr.infer_sig_digits(b))
        return mgr.format_sig(val, sig)

    def div_sig(self,a: str, b: str) -> str:
        val = _to_decimal(a) / _to_decimal(b)
        sig = min(mgr.infer_sig_digits(a), mgr.infer_sig_digits(b))
        return mgr.format_sig(val, sig)
    def common_sum(self, a: str, b: str, dec: int) -> str:
        val = _to_decimal(a) + _to_decimal(b)
        val = mgr.format_dec(val, dec)
        return val

_ALLOWED_FUNCS = {
    # 常用函数（float 计算用）
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "exp": math.exp, "sqrt": math.sqrt, "abs": abs,
    # 对数
    "ln": math.log, "log": math.log, "log10": math.log10,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)

class _SafeEvaluator(ast.NodeVisitor):
    """把受限 AST 转为可计算的 lambda x: float。仅依赖 float 级别函数。"""
    def __init__(self, expr: str):
        # 允许把 ^ 写成幂
        expr = expr.replace("^", "**")
        self._tree = ast.parse(expr, mode="eval")

    def build(self):
        # 预先绑定局部变量，避免闭包里频繁属性查找
        body = self._tree.body
        eval_node = self._eval

        def f(x: float) -> float:
            return float(eval_node(body, float(x)))

        return f

    def _eval(self, node, x):
        if isinstance(node, ast.Constant):  # 数字常量
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("不支持的常量类型")
        if isinstance(node, ast.Name):
            if node.id == "x":
                return float(x)
            if node.id in _ALLOWED_CONSTS:
                return _ALLOWED_CONSTS[node.id]
            raise ValueError(f"不允许的名字: {node.id}")
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
            left = self._eval(node.left, x)
            right = self._eval(node.right, x)
            if isinstance(node.op, ast.Add):  return left + right
            if isinstance(node.op, ast.Sub):  return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div):  return left / right
            if isinstance(node.op, ast.Mod):  return left % right
            if isinstance(node.op, ast.Pow):  return left ** right
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARYOPS):
            val = self._eval(node.operand, x)
            return +val if isinstance(node.op, ast.UAdd) else -val
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("仅支持简单函数名调用")
            name = node.func.id
            if name not in _ALLOWED_FUNCS:
                raise ValueError(f"不允许的函数: {name}")
            args = [self._eval(a, x) for a in node.args]
            if len(args) != 1:
                raise ValueError("目前仅支持一元函数")
            return _ALLOWED_FUNCS[name](args[0])
        raise ValueError("不支持的表达式结构")
class DecimalFormular:
    def _make_callable(self,expr: str):
        """优先用 AST 安全解析；失败就进入“受限 eval”宽松模式。"""
        try:
            return _SafeEvaluator(expr).build(), "ast"
        except Exception:
            # 宽松模式：eval 但只开放白名单
            safe_ns = {}
            safe_ns.update(_ALLOWED_FUNCS)
            safe_ns.update(_ALLOWED_CONSTS)
            code = compile(expr.replace("^", "**"), "<expr>", "eval")
            def f(x: float) -> float:
                return float(eval(code, {"__builtins__": {}}, {**safe_ns, "x": float(x)}))
            return f, "eval"

# ---------------- 主接口：表达式 + x（字符串） -> 有效数字结果 ----------------
    def eval_sig_expr(self, expr: str, x_literal: str) -> list[str]:
        """
        输入：
            expr: 字符串函数表达式，如 "sin(x)+log10(x^2+1)"
            x_literal: x 的字面量，如 "12.30"、"7.10e3"（用于读取 σx 的位权）
        输出：按 σy 的那一位进行银行家舍入后的字符串
        """
        # 1) σx & h
        p_x = _sigma_exp_from_literal(x_literal)
        sigma_x = _exp10(p_x)                # Decimal
        h = float(sigma_x)                   # 差分步长与“最后一位”同位权

        # 2) 构造 f(x)
        f, mode = self._make_callable(expr)

        # 3) y 与导数
        x_dec = _to_decimal(x_literal)
        x = float(x_dec)
        y_val = f(x)
        # 中心差分（异常/不可导点自然由函数值决定）
        try:
            y_p = f(x + h)
            y_m = f(x - h)
            dy = (y_p - y_m) / (2.0 * h)
        except Exception:
            # 如果在端点或奇点附近爆了，缩小步长再试
            h2 = h * 0.1 if h != 0 else 1e-12
            y_p = f(x + h2); y_m = f(x - h2); dy = (y_p - y_m) / (2.0 * h2)

        # 4) σy 与最后一位
        y_dec = _to_decimal(str(y_val))
        dy_dec = _to_decimal(str(dy))
        sigma_y = abs(Decimal(str(dy))) * sigma_x
        if sigma_y.is_zero():
            # 极少数导数≈0 的点：退回显示 1 位小数
            return [mgr.format_dec(y_dec, 1), mgr.format_dec(dy_dec, 1)]
        # 位权 10^p_place
        p_place = int(math.floor(math.log10(float(sigma_y))))
        return [_round_y_to_place(y_dec, p_place), _round_y_to_place(dy_dec, p_place)]

"""
以下是api文档
calc = DecimalCalculator()
x = 698965.48835
y = 96865158.456965
x_str = mgr.format_sig(x,10)
y_str = mgr.format_sig(y,10)
print(calc.sum_sig(x_str, y_str))#+
print(calc.sub_sig(x_str, y_str))#-
print(calc.mul_sig(x_str, y_str))#*
print(calc.div_sig(x_str, y_str))#/

formu = DecimalFormular()
#支持复杂的函数表达式(在map里的都可以，支持表达式)
print(formu.eval_sig_expr("ln(x)", y_str))
"""














