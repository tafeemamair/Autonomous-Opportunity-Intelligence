from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    environment: str = "development"
    log_level: str = "INFO"
    database_url: str = "sqlite:///./aoi.db"
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "aoi"
    tavily_api_key: str | None = None
    exa_api_key: str | None = None
    firecrawl_api_key: str | None = None

    model_config = SettingsConfigDict(env_prefix="AOI_", env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
