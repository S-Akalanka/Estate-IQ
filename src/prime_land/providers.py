from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from prime_land.config import CONFIG, get_api_key

PROVIDER = CONFIG["provider"]


def get_embeddings() -> OpenAIEmbeddings:
    """Create the embedding model served through OpenRouter."""
    return OpenAIEmbeddings(
        model=PROVIDER["embedding_model"],
        openai_api_key=get_api_key(),
        openai_api_base=PROVIDER["openrouter_base_url"],
        chunk_size=PROVIDER["embedding_batch_size"],
        check_embedding_ctx_length=False,
    )


def get_llm() -> ChatOpenAI:
    """Create the chat model served through OpenRouter."""
    return ChatOpenAI(
        model=PROVIDER["chat_model"],
        temperature=PROVIDER["temperature"],
        max_tokens=PROVIDER["max_tokens"],
        openai_api_key=get_api_key(),
        openai_api_base=PROVIDER["openrouter_base_url"],
    )
