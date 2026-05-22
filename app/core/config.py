from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Body Engine"
    app_version: str = "0.1.0"
    environment: str = "development"
    model_dir: str = "./models"
    data_dir: str = "./data"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )


settings = Settings()
