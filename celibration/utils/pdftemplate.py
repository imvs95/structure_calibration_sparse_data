from fpdf import FPDF, HTMLMixin
from pandas import DataFrame


class ReportTemplate(FPDF):
    def __init__(self):
        super().__init__()
        self.add_page()
        self.set_auto_page_break(True, 10)
        self.set_font('Arial', '', 12)
        self.set_top_margin(10)
        self.set_left_margin(10)
        self.set_right_margin(10)

    def report_title(self, title: str):
        self.set_xy(10, 10)
        self.set_font_size(32)
        self.cell(0, 11, title, 0, 1, 'L', False)

    def parameters(self, parameters: str):
        self.set_xy(10, 20)
        self.set_font_size(20)
        self.cell(0, 7, 'Parameters', 0, 1, 'L', False)
        self.set_xy(10, 30)
        self.set_font_size(10)
        self.write(5, parameters)

    def results(self, results):
        if not isinstance(results, str):
            results = str(results)
        self.set_xy(10, self.get_y() + 10)
        self.set_font_size(20)
        self.cell(0, 7, 'Results', 0, 1, 'L', False)
        self.set_xy(10, self.get_y() + 10)
        self.set_font_size(10)
        self.write(5, results)


class SummaryTemplate(FPDF, HTMLMixin):
    def __init__(self):
        super().__init__()
        self.add_page(orientation="L")
        self.set_auto_page_break(True, 10)
        self.set_font('Arial', '', 12)
        self.set_top_margin(10)
        self.set_left_margin(10)
        self.set_right_margin(10)

    def write_summary(self, metadata: DataFrame):
        self.set_xy(10, 10)
        self.set_font_size(32)
        self.cell(0, 11, "Summary", 0, 1, 'L', False)
        self.set_font_size(8)
        html = metadata.to_html(col_space=1, index=False)
        for w in [30, 50, 10, 40, 45, 20]:
            html = html.replace("style=\"min-width: 1px;\"", f"width=\"{w}\"", 1)
        html = html.replace("style=\"min-width: 1px;\"", f"width=\"35\"")
        self.write_html(html)
