import time
import datetime


def now_ms():
    return int(round(time.utctime() * 1000))


def now_ns():
    return int(round(time.utctime() * 1.0e9))


def hex_from_ms(ms):
    return '%12.12x' % ms


def hex_from_ns(ns):
    return '%16.16x' % ns


def ms_from_hex(hex_ms):
    if len(hex_ms) != 12:
        raise Exception('Invalid Millisecond Hex Timestamp: %s' % hex_ms)
    return int(hex_ms, 16)


def ns_from_hex(hex_ns):
    if len(hex_ns) != 16:
        raise Exception('Invalid Nanosecond Hex Timestamp: %s' % hex_ns)
    return int(hex_ns, 16)


def datetime_from_ms(ms):
    return datetime.datetime.utcfromtimestamp(ms / 1000.0)


def datetime_from_ns(ns):
    return datetime.datetime.utcfromtimestamp(ns / 1.0e9)


def datetime_from_hex_ms(hex_ms):
    return datetime_from_ms(ms_from_hex(hex_ms))


def datetime_from_hex_ns(hex_ns):
    return datetime_from_ns(ns_from_hex(hex_ns))


def ms_from_datetime(dt):
    return int(time.mktime(dt.utctimetuple()) * 1000)


def ns_from_datetime(dt):
    return int(time.mktime(dt.utctimetuple()) * 1.0e9)


def hex_ms_from_datetime(dt):
    return hex_from_ms(ms_from_datetime(dt))


def hex_ns_from_datetime(dt):
    return hex_from_ns(ns_from_datetime(dt))
