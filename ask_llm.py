from openai import OpenAI


def get_llm_answer(client: OpenAI, context: str, question: str, model: str = "gpt-4o"):
    # Define system and user prompts
    SYSTEM_PROMPT = """
    你是一个搜索助手，精通鲁迅的文章，如果搜到了相关的文章的片段，你可以告诉用户关于这些片段的信息，和片段的内容。
    """
    USER_PROMPT = f"""
    <context> 标签内是可能的鲁迅的作品片段，<question> 标签内是用户的问题。
    如果用户问题和这个片段相关，那么就回答用户的问题，并给出片段的信息。如果不相关，那么就直接回答用户的问题。
    
    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )

    answer = response.choices[0].message.content
    return answer
