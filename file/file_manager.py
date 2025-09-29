from operator import index
import pandas as pd
class ExcelReader:
    def __init__(self):
        self.path = None
        self.sheet_name = None
        self.header = None
        self.index_col = None
        self.dtype = None
        self.engine = None
        self.excel = {}
    def write_kwargs(self,
    path: str,
    sheet_name = "Sheet1",
    header = None,
    index_col = None,
    dtype: list = None,
    engine: str = None,
    ):
        self.path = path
        self.sheet_name = sheet_name
        self.header = header
        self.index_col = index_col
        self.dtype = dtype
        self.engine = engine
    def read_excel(self, key):
        excel = pd.read_excel(self.path, sheet_name=self.sheet_name, header=self.header,index_col=self.index_col,dtype=self.dtype,engine=self.engine)
        self.excel[key] = excel
        return excel








