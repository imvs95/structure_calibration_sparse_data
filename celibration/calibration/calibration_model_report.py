""" Abstract definition of the building blocks of a CalibrationModelReport object
"""

from typing import Any
from abc import ABC, abstractmethod
from .calibration_model import CalibrationModel
import pandas as pd


class CalibrationModelReport(ABC):
    def __init__(self, model: CalibrationModel, **kwargs):
        self._model = model

    @abstractmethod
    def get_string(self) -> str:
        return NotImplementedError

    @abstractmethod
    def render(self, type: str) -> Any:
        """Abstract skeleton for rendering a CalibrationModelReport

        Args:
            type (str): Render type

        Returns:
            Any: Arbitrary object that can be rendered by the host's machine
        """
        return NotImplementedError

    def export_to_file(self, type: str, filename: str) -> Any:
        """Abstract skeleton for exporting a CalibrationModelReport to a file

        Args:
            type (str): File type
            filename (str): Name of the file to be written to

        Returns:
            Any: Arbitrary electronic document that can be viewed by the host's machine
        """
        return NotImplementedError

    def export_to_pdf(self, filename: str) -> Any:
        from celibration.utils.pdftemplate import ReportTemplate
        import yaml
        p = ReportTemplate()
        p.report_title(self._model.get_info()['name'])
        minfo = self._model.get_info()

        #add to pdf for readability
        model_info = minfo.copy()
        for k, v in model_info["parameters"].items():
            if isinstance(v, dict):
                model_info["parameters"][k] = {"keys": list(v.keys())}
            elif isinstance(v, pd.DataFrame):
                model_info["parameters"][k] = {"len_df": len(v), "columns_df": list(v.columns)}

        par_str = yaml.safe_dump(model_info, allow_unicode=True, default_flow_style=False, indent=4)
        p.parameters(par_str)
        p.results(self._model.get_summary()[1])
        p.output(filename)
