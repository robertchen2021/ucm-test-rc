from .utils import show_current_plot
from matplotlib import pyplot as plt
from nauto_ucm_accountant.accountant.structs import PRCurve, TimeRange


def render_pr_curve(pr_curve: PRCurve, suptitle: str):
    plt.title('Precision VS recall')
    plt.suptitle(suptitle)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.fill_between(pr_curve.recall_values, 0, pr_curve.precision_values, alpha=.3)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.plot(pr_curve.recall_values, pr_curve.precision_values)
    show_current_plot()


def render_f1_threshold_curve(pr_curve: PRCurve, suptitle: str):
    plt.plot(
        pr_curve.threshold_values,
        pr_curve.f1_values,
        markevery=[pr_curve.best_threshold_index],
        marker='D',
        markersize=8,
        markerfacecolor='orange'
    )
    plt.fill_between(pr_curve.threshold_values, 0, pr_curve.f1_values, alpha=.3)
    plt.xlabel('Threshold')
    plt.ylabel('F1 score')
    plt.axvline(pr_curve.best_threshold, color='gray', linestyle='--')
    plt.annotate(
        'Thresh.: {0:.2}\nPrec.: {2:.2}\nRec.: {3:.2}\nF1: {1:.2}'.format(
            pr_curve.best_threshold,
            pr_curve.precision_values[pr_curve.best_threshold_index],
            pr_curve.recall_values[pr_curve.best_threshold_index],
            pr_curve.f1_values[pr_curve.best_threshold_index]
        ),
        xy=(pr_curve.best_threshold * 1.05, .97),
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=11
    )
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.title('F1 score VS threshold')
    plt.suptitle(suptitle)
    show_current_plot()


def render_time_range(time_range: TimeRange, ylabel: str, suptitle: str):
    x = list(range(len(time_range.distribution.recommended_distribution.keys())))
    plt.xlabel('Date')
    plt.xticks(x, time_range.distribution.recommended_distribution.keys(), rotation=45)
    plt.bar(x, time_range.distribution.recommended_distribution.values())
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} per {time_range.distribution.key_name} (from{time_range.min_time} to {time_range.max_time})")
    plt.suptitle(suptitle)
    show_current_plot()
