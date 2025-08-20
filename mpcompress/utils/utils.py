import os
import re
import logging
from datetime import datetime


def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def setup_logger(
    logger_name, root, phase, level=logging.INFO, screen=False, tofile=False
):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, f"{phase}.log")
        fh = logging.FileHandler(log_file, mode="a")
        # fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        # sh.setFormatter(formatter)
        lg.addHandler(sh)


def rename_key_by_rules(key: str, rules: list) -> str:
    """根据规则重命名 key，优先使用预设语义，最后使用regex匹配

    Args:
        key: 要处理的原始键名
        rules: 规则列表，每条规则格式为 [type, pattern, replacement]
               type 可以是: "startswith", "endswith", "contains", "exact", "regex"

    Returns:
        重命名后的键名
    """
    for rule in rules:
        if len(rule) != 3:
            print(f"Warning: Invalid rule format: {rule}")
            continue  # 跳过格式不正确的规则

        rule_type, pattern, replacement = rule

        if rule_type == "startswith":
            if key.startswith(pattern):
                return replacement + key[len(pattern) :]

        elif rule_type == "endswith":
            if key.endswith(pattern):
                return key[: -len(pattern)] + replacement

        elif rule_type == "contains":
            if pattern in key:
                return key.replace(pattern, replacement, 1)  # 只替换第一个匹配

        elif rule_type == "exact":
            if key == pattern:
                return replacement

        elif rule_type == "regex":
            if isinstance(pattern, str):
                if re.match(pattern, key):
                    return re.sub(pattern, replacement, key)
            elif isinstance(pattern, re.Pattern):
                if pattern.match(key):
                    return pattern.sub(replacement, key)

    return key  # 无匹配时返回原 key
