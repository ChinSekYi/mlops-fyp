import os

import yaml
from dotenv import load_dotenv


def load_environment(env_file: str = None):
    """
    Load environment variables from a file.
    Default is .env at project root.
    """
    if env_file is None:
        env_file = ".env"

    load_dotenv(env_file)
    print(f"[ENV] Loaded environment from {env_file}")
