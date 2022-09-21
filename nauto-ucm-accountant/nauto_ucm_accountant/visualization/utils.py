import io
from matplotlib import pyplot as plt
from contextlib import contextmanager


def show_current_plot(full_screen=False):
    if full_screen:
        img = io.StringIO()
        plt.show()
        plt.savefig(img, format='svg', bbox_inches='tight')
        img.seek(0)
        print("%html <div style='width: 800px; height: 2200px;'>" + img.getvalue() + "</div>")

    else:
        plt.show()
        with import_zeppelin() as z:
            z.showplot(plt)
    plt.clf()
    plt.cla()
    plt.close()


@contextmanager
def import_zeppelin():
    from nauto_ucm_accountant.visualization.notebook import zeppelin_context
    if zeppelin_context is not None:
        yield zeppelin_context
    else:
        yield type("", (), {
            "__getattr__": (lambda self, method_name: lambda *args, **kwargs: None)
        })()


def is_zeppelin():
    from nauto_ucm_accountant.visualization.notebook import zeppelin_context
    return zeppelin_context is not None
