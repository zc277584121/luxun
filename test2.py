
from pymilvus import MilvusClient
from pymilvus import DataType


if __name__ == '__main__':
    client = MilvusClient(uri='./luxun2.db');
    res = client.has_collection('LuXunWorks')
    collection_name = "LuXunWorks"
    #print(res)
    schema = MilvusClient.create_schema(auto_id=False)
        # 添加字段到schema
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=256)
    schema.add_field(field_name="book", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="type", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="date", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="chunk", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="window", datatype=DataType.VARCHAR, default_value="", max_length=6144)
    schema.add_field(field_name="method", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="dense_vectors", datatype=DataType.FLOAT_VECTOR, dim=512)

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )

