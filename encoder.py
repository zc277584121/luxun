from openai import OpenAI


def emb_text(client: OpenAI, text: str, model: str = "text-embedding-3-small"):
    embedding = client.embeddings.create(input=text, model=model).data[0].embedding
    return embedding

def emb_batch_texts(client: OpenAI, texts: list, model: str = "text-embedding-3-small"):
    res = client.embeddings.create(input=texts, model=model)
    res = [r.embedding for r in res.data]
    return res
