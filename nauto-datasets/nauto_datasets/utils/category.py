
class Monoid:
    """
    Not an absetract class to avoid metaclass clashes
    """
    @staticmethod
    def zero() -> 'Monoid':
        raise NotImplementedError()

    @staticmethod
    def add(m1: 'Monoid', m2: 'Monoid') -> 'Monoid':
        raise NotImplementedError()


class DummyMonoid(Monoid):
    @staticmethod
    def zero() -> 'DummyMonoid':
        return DummyMonoid()

    @staticmethod
    def add(m1: 'DummyMonoid', m2: 'DummyMonoid') -> 'DummyMonoid':
        return m1
