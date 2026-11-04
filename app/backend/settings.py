"""Backend settings, read from the environment and .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ZARAI_", env_file=".env", extra="ignore")

    app_config: str = "configs/app.yaml"
    cors_origins: list[str] = ["http://localhost:5173"]
    max_upload_mb: int = 10


settings = Settings()
