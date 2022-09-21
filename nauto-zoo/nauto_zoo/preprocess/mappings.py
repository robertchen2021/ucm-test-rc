from typing import NamedTuple


SOURCE_TYPE_CONSTANT = 'constant'
SOURCE_TYPE_MEDIA = 'media'


class Mapping(NamedTuple):
    source: str
    source_type: str
    target: str
