import os
from pathlib import Path

# install.py 所在目录
project_root = Path(__file__).resolve().parent

# .env 文件路径（与 install.py 同目录）
env_path = project_root / ".env"

env_data = {"PROJECT_ROOT": str(project_root)}

if not env_path.exists():
    with env_path.open("w", encoding="utf-8") as f:
        for k, v in env_data.items():
            f.write(f"{k}={v}\n")
