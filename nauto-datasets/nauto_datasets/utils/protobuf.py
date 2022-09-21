import gzip
from pathlib import Path
from typing import Union

from google.protobuf import text_format
from google.protobuf.message import Message


def parse_message_from_bytes(proto_object_t: type, msg_data: bytes) -> Message:
    """Builds a protobuf `Message` from bytes

    Args:
        proto_object_t: a type of the class to be materialized
        msg_data: bytes of serialized message

    Returns:
        parsed and materialized `Message`
    """
    instance = proto_object_t()
    instance.ParseFromString(msg_data)
    return instance


def parse_message_from_gzipped_bytes(proto_object_t: type,
                                     msg_gzip_data: bytes) -> Message:
    """Builds a protobuf `Message` from gzipped bytes

    Args:
        proto_object_t: a type of the class to be materialized
        msg_gzip_data: compressed representation of serialized proto

    Returns:
        parsed and materialized `Message`

    Raises:
        OSError: when msg_gzip_data could not be decompressed
    """
    return parse_message_from_bytes(proto_object_t,
                                    gzip.decompress(msg_gzip_data))


def parse_message_from_file(proto_object_t: type, file_path: Path) -> Message:
    """Builds a protobuf `Message` from file

    Args:
        proto_object_t: a type of the class to be materialized
        file_path: path to serialized proto `Message`

    Returns:
        parsed and materialized `Message`

    Raises:
        FileNotFoundError: when file_path does not exist
    """
    with file_path.open('rb') as f:
        return parse_message_from_bytes(proto_object_t, f.read())


def parse_message_from_gzipped_file(proto_object_t: type,
                                    file_path: Path) -> Message:
    """Builds a protobuf `Message` from gzipped file

    Args:
        proto_object_t: a type of the class to be materialized
        file_path: path to serialized and compressed proto `Message`

    Returns:
        parsed and materialized `Message`

    Raises:
        FileNotFoundError: when file_path does not exist
        OSError: when the file could not be decompressed
    """
    with gzip.open(str(file_path), 'rb') as f:
        return parse_message_from_bytes(proto_object_t, f.read())


def parse_message_from_txt(
        proto_object_t: type, txt_msg: Union[str, bytes]) -> Message:
    """Builds a protobuf `Message` from textual representation (pbtxt)

    Args:
        proto_object_t: a type of the class to be materialized
        txt_msg: textual representation of the object

    Returns:
        an instance of proto_object_t
    """
    msg = proto_object_t()
    text_format.Parse(txt_msg,  msg)
    return msg


def message_to_txt(msg: Message) -> str:
    """Writes a protobuf `Message` to textual representation

    Args:
        msg: an instance of protobuf `Message`

    Returns:
        a string representing the serialized `msg`
    """
    return text_format.MessageToString(msg)
