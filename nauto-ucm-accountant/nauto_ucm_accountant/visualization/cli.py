from typing import List
from nauto_ucm_accountant.accountant.structs import ModelBooks, ModelStatistics
from functools import partial


def render_model_selector(models: List[ModelBooks]) -> str:
    available_models = [m.model_name for m in models]
    print("Available models:")
    for model_index in range(len(available_models)):
        print(f"{model_index}: {available_models[model_index]}")
    selected_model_index = None
    while selected_model_index not in range(len(available_models)):
        selected_model_index = int(input(f"Please select model (0-{len(available_models) - 1}): "))
    print(f"Selected model: {available_models[selected_model_index]}")
    return available_models[selected_model_index]


def render_model_version_selector(model: ModelBooks) -> str:
    print("Available versions:")
    for model_version_index in range(len(model.versions)):
        print(f"{model_version_index}: {model.versions[model_version_index]}")
    selected_model_version_index = None
    while selected_model_version_index not in range(len(model.versions)):
        answer = input(f"Please select model version (0-{len(model.versions) - 1}) or press ENTER for overall analysis: ")
        if answer == "":
            print("No version selected")
            return answer
        elif answer not in [str(opt) for opt in range(len(model.versions))]:
            print("Invalid selection")
        else:
            selected_model_version_index = int(answer)
    print(f"Selected version: {model.versions[selected_model_version_index]}")
    return model.versions[selected_model_version_index]


def render_overview_start():
    columns = ["model", "version", "F1", "Precision", "Recall", "Accuracy", "Coverage", "TP", "FP", "TN", "FN", "Total"]
    _justify = partial(justify, cell_size=10)
    print()
    print("|".join([_justify(c).upper() for c in columns]) + "|")


def render_overview_row(model_name: str, model_version: str, statistics: ModelStatistics):
    columns = [
        model_name,
        model_version,
        f"{statistics.pr_data.f1 * 100:.2f}%",
        f"{statistics.pr_data.precision * 100:.2f}%",
        f"{statistics.pr_data.recall * 100:.2f}%",
        f"{statistics.pr_data.accuracy * 100:.2f}%",
        f"{statistics.coverage * 100:.2f}%",
        statistics.pr_data.true_positives,
        statistics.pr_data.false_positives,
        statistics.pr_data.true_negatives,
        statistics.pr_data.false_negatives,
        statistics.pr_data.num_cases
    ]
    _justify = partial(justify, cell_size=10)
    print("|".join([_justify(str(c)) for c in columns]) + "|")


def justify(text: str, cell_size: int):
    text_size = cell_size - 2
    if len(text) <= text_size:
        text = text.ljust(text_size, " ")
    else:
        text = text[0:text_size-1] + "."
    return f" {text} "
