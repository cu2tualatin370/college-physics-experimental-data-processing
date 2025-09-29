import pandas as pd
from error.error_manager import DataProcessor
from my_decimal.decimal_manager import SignificantDigitsManager,_to_decimal, RoundRule
from my_decimal.decimal_calculator import DecimalCalculator, DecimalFormular
from my_decimal.decimal_calculator import _place_exp_from_sigma, _round_y_to_place
from file.file_manager import ExcelReader
import parameter
from typing import Iterable, Tuple, Any, Literal
mgr = SignificantDigitsManager()
calc = DecimalCalculator()
form = DecimalFormular()
exc = ExcelReader()
pro = DataProcessor()

#进行处理,键入你想处理的块
def format_mean_with_sigma(mean_literal: str, sigma_literal: str) -> str:
    """
    参数：
        mean_literal  : 平均值的“字面量字符串”（如 "12.3049" 或 "1.230e3"）
        sigma_literal : 不确定度 σ 的“字面量字符串”（如 "0.0567" 或 "5.67e-2"）
    返回：
        (mean_fmt, sigma_fmt) 两个字符串，平均值最后一位与 σ 的位权对齐，且按银行家舍入。
    """
    y = _to_decimal(mean_literal)     # 稳健转 Decimal（避免 float 误差）
    s = _to_decimal(sigma_literal)

    if s.is_zero():
        # σ=0 的极端情况：降级策略（这里给出示例：保留 1 位小数）
        return mgr.format_dec(y, 1)

    # 找出 σ 的“位权指数” p_place（等价 floor(log10(σ))）
    p_place = _place_exp_from_sigma(s)

    # 把 y 和 σ 都四舍六入五凑偶到这一位权
    y_fmt = _round_y_to_place(y, p_place)
    return y_fmt
def process(df: pd.DataFrame, rows:Tuple[Any, Any], arrs:Tuple[Any, Any], device_deviation:float):
    block = pro.rect(df,rows=rows,cols=arrs,dec_places=parameter.dec_all, by="position")
    block = pro.format_dec_batch(block, dec=parameter.dec_all)
    average = pro.average(block)
    a_deviation = pro.standard_deviation(block,average)
    out = pro.combined_uncertainty(a_deviation,device_deviation,parameter.device_sig)
    combined_deviation = out[0]
    b_deviation = out[1]
    ex = pro.relative_uncertainty(combined_deviation,average)
    combined_deviation_out = mgr.format_sig(combined_deviation,parameter.combined_deviation_out_sig,rule= RoundRule.UP)
    average_out = format_mean_with_sigma(average, combined_deviation_out)
    ex_out = mgr.format_sig(ex, parameter.ex_out_sig, rule= RoundRule.UP)
    print(f"average:{average}")
    print(f"combined_deviation:{combined_deviation}")
    print(f"ex:{ex}")
    print(f"average_out:{average_out}")
    print(f"a_deviation:{a_deviation}")
    print(f"b_deviation:{b_deviation}")
    print(f"combined_deviation_out:{combined_deviation_out}")
    print(f"ex_out:{ex_out}")
#读取表格
exc.write_kwargs(path=r"D:\ra2ol\code\PythonProject8\study1.xlsx", sheet_name="Sheet1")
df = exc.read_excel(0)
process(df,rows=(0,0),arrs=(0,4),device_deviation=0.01)

