import glob
import json
import argparse
from time import time, sleep
import sys
import os
import re
import logging
from pathlib import Path, PurePath
from datetime import datetime
from PyPDF3 import PdfFileMerger, PdfFileReader
from celibration import CalibrationPlanManager
from typing import Any

# Couple these to input parameters
dt = datetime.now()
ts = datetime.timestamp(dt)
reports_dt = dt.strftime("%Y%m%d_%H%M%S")
reporting_enabled = False


def merge_pdf(reports_folder: Any, filename_out: str):
    # Merge pdf file
    merger = PdfFileMerger()
    pdfs = glob.glob(str(reports_folder / "*.pdf"))
    # Move summary to the end of the list
    r = re.compile(".*_summary.pdf")
    if any((m := r.match(x)) for x in pdfs):
        pdfs.append(pdfs.pop(pdfs.index(m.group(0))))

    for fn in pdfs:
        filepath = str(reports_folder / fn)
        if os.path.isfile(filepath):
            merger.append(PdfFileReader(open(filepath, "rb")))
    merger.write(filename_out)


def print_report(**kwargs):
    plan = kwargs["plan"]
    plan_state = plan.get_state()
    models = plan_state["calibration_models"]

    logging.info(f"******************** Reports ********************")
    for m in models:
        model_info = m.get_info()
        diff_func = m.diff_func_object
        extended_report = (f"{m.get_report().get_string()}").expandtabs(2)

        reports_folder = (
            Path("reports") / Path(reports_dt + "_" + plan.yaml_name)
        ).absolute()
        os.makedirs(reports_folder, exist_ok=True)

        if reporting_enabled:
            # Save json report
            filename = Path(reports_folder) / (
                model_info["cm_name"] + "_" + diff_func.get_info()["name"] + ".json"
            )
            m.get_report().export_to_file(type="json", filename=str(filename))

            # Save pdf report
            filename = Path(reports_folder) / (
                model_info["cm_name"] + "_" + diff_func.get_info()["name"] + ".pdf"
            )
            m.get_report().export_to_file(type="pdf", filename=str(filename))
        print(extended_report)


def run(args):
    global reporting_enabled
    reporting_enabled = args.report if args.report else reporting_enabled
    plan_yaml_path = PurePath(args.plan)
    yaml_name = str(plan_yaml_path.name).split(".yml")[0]
    reports_folder = (Path("reports") / Path(reports_dt + "_" + yaml_name)).absolute()

    manager = CalibrationPlanManager(config="./celibration/celibration_config.yaml")
    if args.verbosity:
        manager.set_verbosity(int(args.verbosity))
    manager.read_plan_from_yaml(filename=args.plan)
    manager.onPlanUpdated += print_report
    manager.run(args.multiprocessing)

    metadata = manager._metadata
    total_json = ""
    # JSON comparison summaries
    for plan_name, meta_df in metadata.items():
        meta_df = meta_df.reset_index(drop=True)
        meta_json = meta_df.to_json(orient="table")
        total_json += meta_json

        with open(
            str(reports_folder / f"{plan_name}_summary.json"), "w+", encoding="utf-8"
        ) as f:
            f.write(json.dumps(json.loads(meta_json), indent=4))

    with open(
        str(reports_folder / "plan_summaries_merged.json"), "w+", encoding="utf-8"
    ) as f:
        f.write(json.dumps(json.loads(total_json), indent=4))

    # Wait for prior events to be handled
    sleep(2)
    if reporting_enabled:
        from celibration.utils.pdftemplate import SummaryTemplate

        for plan_name, meta_df in metadata.items():
            template = SummaryTemplate()
            template.write_summary(meta_df)
            template.output(str(reports_folder / f"{plan_name}_summary.pdf"))
        merge_pdf(
            reports_folder=reports_folder,
            filename_out=str(reports_folder / "reports_merged.pdf"),
        )


def main():
    parser = argparse.ArgumentParser(
        prog="celibration",
        description="A generic architecture to calibrate and compare optimization and ML models",
    )
    parser.add_argument(
        "-p",
        "--plan",
        default="tests/example.yml",
        help="Provide your yml plan here",
        required=False,
    )
    parser.add_argument(
        "-pfs",
        "--plugins_folders",
        nargs="+",
        help="Provide your custom plugin folders separated with a space in case there are more",
        required=False,
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        help="increase output verbosity",
        choices=["0", "1", "2"],
        default="0",
        required=False,
    )

    parser.add_argument(
        "-r", "--report", dest="report", help="Report output type", action="store_true"
    )
    parser.add_argument(
        "-mp", "--enable-multiprocessing", dest="multiprocessing", action="store_true"
    )

    parser.set_defaults(multiprocessing=False)
    parser.set_defaults(report=False)
    args = parser.parse_args()
    run(args)


import pandas as pd
from logger import Logger

if __name__ == "__main__":
    # log = logging.getLogger()
    # handler = logging.FileHandler(filename='C:/Users/IvS/Git/kalibratietechnieken/log_test.log')
    # log.addHandler(handler)

    #Logger("log_test_2.txt")

    pd.set_option("display.width", 400)
    pd.set_option("display.max_columns", 10)
    main()
