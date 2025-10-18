import os

from dotenv import load_dotenv


def load_environment(env_file: str = None):
    """
    Load environment variables from a file in /env directory.
    Default is .env at project root's /env folder.
    """
    project_root = os.path.dirname(os.path.dirname(__file__))
    env_dir = os.path.join(project_root, "env")
    if env_file is None:
        env_file = "env/.env"
    env_path = (
        os.path.join(project_root, env_file)
        if not os.path.isabs(env_file)
        else env_file
    )
    if not os.path.exists(env_path):
        env_path = os.path.join(env_dir, ".env")
    load_dotenv(env_path)
    print(f"[ENV] Loaded environment from {env_path}")
