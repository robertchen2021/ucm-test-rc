from typing import List, Optional, Union
from nauto_ucm_accountant.accountant.structs import ModelBooks, ModelVersionBooks
from nauto_ucm_accountant.visualization import plt, notebook, cli  # todo bokeh
from .utils import is_zeppelin


Books = Union[ModelBooks, ModelVersionBooks]


def render_model_selector(models: List[ModelBooks]) -> Optional[str]:
    return get_env().render_model_selector(models)


def render_model_version_selector(model: ModelBooks) -> Optional[str]:
    return get_env().render_model_version_selector(model)


def render_events_hist(books: Books):
    plt.render_time_range(books.statistics.time_range, "Num events", books.identifier)


def render_pr_curve(books: Books):
    plt.render_pr_curve(books.statistics.pr_curve, books.identifier)


def render_f1_threshold_curve(books: Books):
    plt.render_f1_threshold_curve(books.statistics.pr_curve, books.identifier)


def render_coverage_and_pr(model_books: ModelBooks, model_versions_books: List[ModelVersionBooks]):
    get_env().render_overview_start()
    for v in model_versions_books:
        get_env().render_overview_row(v.model_name, v.version, v.statistics)
    get_env().render_overview_row(model_books.model_name, "", model_books.statistics)


def get_env():
    return notebook if is_zeppelin() else cli  # todo use objects to enforce interface
