from typing import List
from nauto_ucm_accountant.accountant.structs import ModelBooks, ModelStatistics
from .utils import import_zeppelin


zeppelin_context = None


def initialize(z):
    # todo use objects instead of global vars
    global zeppelin_context
    zeppelin_context = z


def render_model_selector(models: List[ModelBooks]) -> str:
    with import_zeppelin() as z:
        return z.select(
            "Please select model",
            [(m.model_name, m.model_name) for m in models]
        )


def render_model_version_selector(model: ModelBooks) -> str:
    no_version = ('', '')
    with import_zeppelin() as z:
        return z.select(
            "(optional) Please select model version",
            [no_version] + [(v, v) for v in model.versions]
        )


def render_overview_start():
    print("%table "
          "Model"
          "\tVersion"
          "\tF1 score"
          "\tPrecision"
          "\tRecall"
          "\tAccuracy"
          "\tCoverage"
          "\tTrue positives"
          "\tFalse positives"
          "\tTrue negatives"
          "\tFalse negatives"
          "\tTotal cases")


def render_overview_row(model_name: str, model_version: str, statistics: ModelStatistics):
    print(f"{model_name}"
          f"\t{model_version}"
          f"\t{statistics.pr_data.f1 * 100:.2f}%"
          f"\t{statistics.pr_data.precision * 100:.2f}%"
          f"\t{statistics.pr_data.recall * 100:.2f}%"
          f"\t{statistics.pr_data.accuracy * 100:.2f}%"
          f"\t{statistics.coverage * 100:.2f}%"
          f"\t{statistics.pr_data.true_positives}"
          f"\t{statistics.pr_data.false_positives}"
          f"\t{statistics.pr_data.true_negatives}"
          f"\t{statistics.pr_data.false_negatives}"
          f"\t{statistics.pr_data.num_cases}")
