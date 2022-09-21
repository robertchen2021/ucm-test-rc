"""
This script reads UCM accountant data and displays it locally via CLI
"""
from nauto_ucm_accountant.accountant import AccountantReaderS3
from nauto_ucm_accountant.visualization import render_f1_threshold_curve, render_pr_curve, render_events_hist, render_coverage_and_pr, render_model_selector, render_model_version_selector
from nauto_ucm_accountant.logger import logger


def main():
    logger.info("Reading data...")
    accountant = AccountantReaderS3()

    logger.info("Plotting...")
    model_name = render_model_selector(accountant.get_models())
    model_version_name = render_model_version_selector(accountant.get_model_books(model_name))

    model_books = accountant.get_model_books(model_name)
    version_books = {
        model_version: accountant.get_model_version_books(model_name, model_version)
        for model_version in model_books.versions
    }

    if model_version_name == "":
        selected_books = model_books
    else:
        selected_books = version_books[model_version_name]

    render_coverage_and_pr(model_books, list(version_books.values()))
    render_f1_threshold_curve(selected_books)
    render_pr_curve(selected_books)
    render_events_hist(selected_books)


if __name__ == "__main__":
    main()
