# from openai import OpenAI


# def emb_text(client: OpenAI, text: str, model: str = "text-embedding-3-small"):
#     embedding = client.embeddings.create(input=text, model=model).data[0].embedding
#     return embedding

# def emb_batch_texts(client: OpenAI, texts: list, model: str = "text-embedding-3-small"):
#     res = client.embeddings.create(input=texts, model=model)
#     res = [r.embedding for r in res.data]
#     return res


############################################################
# from pymilvus import model
# import numpy as np
#
# embedding_model = model.DefaultEmbeddingFunction()
#
# def emb_text(text: str):
#     return embedding_model.encode_queries([text])[0]
#
# def emb_batch_texts(texts: list):
#     embeddings = embedding_model.encode_documents(texts)
#     if isinstance(embeddings[0], np.ndarray):
#         return [embedding.tolist() for embedding in embeddings]
#     else:
#         return embeddings
#
#
############################################################

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

def emb_text(text: str) -> list[float]:
    return model.encode([text], normalize_embeddings=True)[0]

def emb_batch_texts(texts: list) -> list[list[float]]:
    res = model.encode(texts, normalize_embeddings=True)
    return res.tolist()


if __name__ == '__main__':
    texts = ['——应《京报副刊》的征求',
             '十月十二日',
             '枣树', ]
    res = emb_batch_texts(texts)
    for r in res:
        print(r[:10])