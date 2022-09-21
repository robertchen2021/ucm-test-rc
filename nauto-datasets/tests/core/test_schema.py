import unittest

from pyspark.sql import types as dtype

from nauto_datasets.core.schema import RecordSchema


class RecordSchemaTests(unittest.TestCase):
    @staticmethod
    def _get_schema() -> RecordSchema:
        return RecordSchema(
            entities=dict(
                first=[
                    dtype.StructField('firstBoolCol', dtype.BooleanType(), False),
                    dtype.StructField('firstDecCol', dtype.DecimalType(), False),
                    dtype.StructField('firstStringCol', dtype.StringType(), True)
                ],
                second=[
                    dtype.StructField('secondBoolCol', dtype.BooleanType(), False),
                    dtype.StructField('secondIntCol', dtype.IntegerType(), False),
                    dtype.StructField(
                        'secondArrayCol',
                        dtype.ArrayType(
                            elementType=dtype.LongType(), containsNull=False),
                        True)
                ],
            ))

    def test_combined_schema(self):
        schema = self._get_schema()

        self.assertEqual(
            schema.combined_schema(),
            dtype.StructType(fields=[
                dtype.StructField('firstBoolCol', dtype.BooleanType(), False),
                dtype.StructField('firstDecCol', dtype.DecimalType(), False),
                dtype.StructField('firstStringCol', dtype.StringType(), True),
                dtype.StructField('secondBoolCol', dtype.BooleanType(), False),
                dtype.StructField('secondIntCol', dtype.IntegerType(), False),
                dtype.StructField(
                    'secondArrayCol',
                    dtype.ArrayType(
                        elementType=dtype.LongType(), containsNull=False),
                    True)
            ]))

    def test_entity_schemas(self):
        schema = self._get_schema()

        self.assertDictEqual(
            schema.entity_schemas(),
            dict(
                first=dtype.StructType(fields=[
                    dtype.StructField('firstBoolCol', dtype.BooleanType(),
                                      False),
                    dtype.StructField('firstDecCol', dtype.DecimalType(),
                                      False),
                    dtype.StructField('firstStringCol', dtype.StringType(),
                                      True)
                ]),
                second=dtype.StructType(fields=[
                    dtype.StructField('secondBoolCol', dtype.BooleanType(),
                                      False),
                    dtype.StructField('secondIntCol', dtype.IntegerType(),
                                      False),
                    dtype.StructField(
                        'secondArrayCol',
                        dtype.ArrayType(
                            elementType=dtype.LongType(), containsNull=False),
                        True)
                ])))

    def test_schema_serialization(self):
        schema = self._get_schema()
        pb_msg = schema.to_pb()
        des_schema = RecordSchema.from_pb(pb_msg)
        self.assertDictEqual(schema.entities, des_schema.entities)
