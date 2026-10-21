from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from prime_land.config import CONFIG, get_api_key

PROVIDER = CONFIG["provider"]


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create the Gemini embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model=PROVIDER["embedding_model"],
        google_api_key=get_api_key(),
        output_dimensionality=PROVIDER["embedding_dimensions"],
    )


def get_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model."""
    return ChatGoogleGenerativeAI(
        model=PROVIDER["chat_model"],
        temperature=PROVIDER["temperature"],
        max_output_tokens=PROVIDER["max_tokens"],
        google_api_key=get_api_key(),
    )
