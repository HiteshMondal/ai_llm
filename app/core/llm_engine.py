from langchain.llms.base import BaseLLM
from langchain.chat_models.base import BaseChatModel
from app.utils.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()


def get_llm() -> BaseChatModel | BaseLLM:
    """Return the configured LLM."""
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        log.info(f"Using Ollama LLM: {settings.llm_model}")
        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        log.info(f"Using OpenAI LLM: {settings.llm_model}")
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    elif provider == "huggingface":
        from langchain_community.llms import HuggingFaceHub
        log.info(f"Using HuggingFace LLM: {settings.llm_model}")
        return HuggingFaceHub(
            repo_id=settings.llm_model,
            huggingfacehub_api_token=settings.huggingface_api_key,
            model_kwargs={"temperature": settings.llm_temperature, "max_new_tokens": settings.llm_max_tokens},
        )

    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'")