from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    environment: str = "development"
    log_level: str = "INFO"
    database_url: str = "sqlite:///./aoi.db"
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "aoi"
    tavily_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AOI_TAVILY_API_KEY", "TAVILY_API_KEY"),
    )
    exa_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AOI_EXA_API_KEY", "EXA_API_KEY"),
    )
    firecrawl_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AOI_FIRECRAWL_API_KEY", "FIRECRAWL_API_KEY"),
    )

    model_config = SettingsConfigDict(env_prefix="AOI_", env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
