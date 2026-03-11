from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    postgres_user: str = "telegram"
    postgres_password: str = "telegram"
    postgres_db: str = "telegram_rag"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    rabbitmq_user: str = "telegram"
    rabbitmq_password: str = "telegram"
    rabbitmq_host: str = "rabbitmq"
    rabbitmq_port: int = 5672
    rabbitmq_vhost: str = "/"

    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_internal_token: str = "change_me"

    telegram_api_id: int = 0
    telegram_api_hash: str = ""
    telegram_bot_token: str = ""
    telegram_session_name: str = "telegram_bot"

    llm_provider: str = "dummy"
    llm_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = "gpt-5"
    openai_embedding_model: str = "text-embedding-3-large"
    llm_system_prompt: str = (
        "You are a question answering assistant.\n\n"
        "Use ONLY the information provided in the context below.\n\n"
        "If the answer is not contained in the context, respond with:\n"
        "\"I could not find the answer in the provided documents.\"\n\n"
        "Do NOT use external knowledge.\n"
        "Do NOT make assumptions.\n"
        "Keep the answer concise.\n\n"
        "Context:\n"
        "{context}\n\n"
        "Question:\n"
        "{question}"
    )
    vector_score_threshold: float = 0.35
    llm_temperature: float = 0.0
    llm_max_tokens: int = 120
    llm_max_answer_words: int = 40
    rag_max_context_chunks: int = 2

    @property
    def postgres_dsn(self) -> str:
        return (
            f"host={self.postgres_host} port={self.postgres_port} "
            f"dbname={self.postgres_db} user={self.postgres_user} password={self.postgres_password}"
        )

    @property
    def rabbitmq_url(self) -> str:
        if self.rabbitmq_vhost == "/":
            vhost_path = "//"
        elif self.rabbitmq_vhost.startswith("/"):
            vhost_path = self.rabbitmq_vhost
        else:
            vhost_path = f"/{self.rabbitmq_vhost}"

        return (
            f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}"
            f"@{self.rabbitmq_host}:{self.rabbitmq_port}{vhost_path}"
        )

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
