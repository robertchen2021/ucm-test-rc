import json
from typing import Dict, List, NamedTuple

from pyspark.sql.types import StructField, StructType

from nauto_datasets.protos import schema_pb2


class RecordSchema(NamedTuple):
    """Schema represents logical groups of columns constituting
    dataset records.

    Groups must not have common columns. The order of columns in each
    group is irrelevant

    Attributes:
        entities: a dictionary mapping a group/entity name to a list
            of columns descriptions. Each description comes in the
            form of Spark's `StructField`
    """
    entities: Dict[str, List[StructField]]

    def combined_schema(self) -> StructType:
        """Returns a row schema combining all the columns from entities into a
        single row StructType
        """
        return StructType(fields=[
            cd for columns in self.entities.values()
            for cd in columns
        ])

    def entity_schemas(self) -> Dict[str, StructType]:
        """Returns a row schema in the form of StructField for each entity"""
        return {
            entity_name:
            StructType(fields=[cd for cd in entity_columns])
            for entity_name, entity_columns in self.entities.items()
        }

    def to_pb(self) -> schema_pb2.RecordSchema:
        """Serializes the schema as protobuf `Message` """
        entities_pb = {
            name: schema_pb2.ColumnDescriptionList(
                column_descriptions=[
                    schema_pb2.ColumnDescription(json=col.json())
                    for col in col_list])
            for name, col_list in self.entities.items()
        }
        return schema_pb2.RecordSchema(entities=entities_pb)

    @staticmethod
    def from_pb(rs_pb: schema_pb2.RecordSchema) -> 'RecordSchema':
        """Deserialzes the schema from protobuf `Message` """
        return RecordSchema(
            {
                name: [StructField.fromJson(json.loads(col_desc.json))
                       for col_desc in cdl.column_descriptions]
                for name, cdl in rs_pb.entities.items()
            })

    @staticmethod
    def pb_message_type() -> type:
        """Returns the type of the associate protobuf `Message`"""
        return schema_pb2.RecordSchema
