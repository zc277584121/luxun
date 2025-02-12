import os
# import ssl
# import certifi
import json
from glob import glob
from tqdm import tqdm

from encoder import emb_batch_texts, emb_text, OpenAI
from milvus_utils import get_milvus_client, create_collection


COLLECTION_NAME = "LuXunWorks"
MILVUS_ENDPOINT = "./milvus.db"


def get_text(data_dir):
    """Load documents and split each into chunks.

    Return:
        A list of dictionary
    """
    text_dicts = []
    for file_path in glob(os.path.join(data_dir, "**/*.json"), recursive=True):
        text_dict = json.load(open(file_path, "r"))
        text_dicts.extend(text_dict)
    return text_dicts


# Get clients
milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT)
openai_client = OpenAI()

# Set SSL context
# ssl_context = ssl.create_default_context(cafile=certifi.where())

# Get text data from data directory
data_dir = "./data"
text_dicts = get_text(data_dir)

# Create collection
dim = len(emb_text(openai_client, "test"))
create_collection(milvus_client=milvus_client, collection_name=COLLECTION_NAME, dim=dim)

# Insert data
data = []
count = 0
batch_size = 256
batched_text_dicts = [text_dicts[i:i + batch_size] for i in range(0, len(text_dicts), batch_size)]
for batch_text_dicts in tqdm(batched_text_dicts, desc="Creating embeddings"):
    batch_windows = [text_dict["window"] for text_dict in batch_text_dicts]
    vectors = emb_batch_texts(openai_client, batch_windows)
    for text_dict, vector in zip(batch_text_dicts, vectors):
        text_dict["vector"] = vector
        chunk_id = text_dict.pop("id")
        text_dict["chunk_id"] = chunk_id
    mr = milvus_client.insert(collection_name=COLLECTION_NAME, data=batch_text_dicts)
    print("Total number of entities/chunks inserted:", mr["insert_count"])
    # data.append(text_dicts)
# print("Total number of loaded documents:", count)

# Insert data into Milvus collection

