import typing


def recreate(cls, values):
    return cls(**values)


class NamedTupleRecreator:
    """Mixin adding different behaviour when unpickling NamedTuple
    derivatives.

    Custom `__reduce__` method prevents from using NamedTuple's original
    overriden `__ruduce__` method, which ignores subclasses.
    """

    def __reduce__(self):
        return recreate, (self.__class__, self._asdict())


class NamedTupleMetaEx(typing.NamedTupleMeta):
    """This is intendent to use a metaclass making it possible
       to use create NamedTuples deriving from other interfaces too.

    Example:
    ``
    class ExTuple(AddedMixin, metaclass=NamedTupleMetaEx):
        a: int
        b: str

    ExTuple(1, 'ala').added_method('crazy stuff')
    ``
    """

    def __new__(cls, typename, bases, ns):
        cls_obj = super().__new__(cls, typename + 'Base', bases, ns)
        bases = (NamedTupleRecreator, cls_obj) + bases
        new_t = type(typename, bases, {})
        # set module back to original class's module
        # otherwise it's full name will be 'nauto_datasets.utils.tuples.typename',
        # what messes up pickling
        new_t.__module__ = ns['__module__']
        return new_t
