import environ
from dotenv import load_dotenv

@environ.config(prefix="")
class AppConfig:
    ollama_host: str = environ.var(default="localhost")
    ollama_port: str = environ.var(default="11434")
    ollama_model: str = environ.var(default="qwen3:8b")

load_dotenv()
config = environ.to_config(AppConfig)
