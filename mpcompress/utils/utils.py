import os
import re
import logging
from datetime import datetime


def get_timestamp():
    """
    Generate a timestamp string in the format YYMMDD-HHMMSS.

    Returns:
        timestamp (str): Timestamp string in format "YYMMDD-HHMMSS".
    """
    return datetime.now().strftime("%y%m%d-%H%M%S")


def setup_logger(
    logger_name, root, phase, level=logging.INFO, screen=False, tofile=False
):
    """
    Set up a logger with optional file and screen handlers.

    Configures a logger with the specified name and level. Can optionally
    add a file handler (appending to log file) and/or a stream handler
    (outputting to console).

    Args:
        logger_name (str): Name of the logger to create or retrieve.
        root (str): Root directory path for log files.
        phase (str): Phase name used to construct log file name (e.g., "train", "test").
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG). Defaults to logging.INFO.
        screen (bool): If True, add a StreamHandler for console output. Defaults to False.
        tofile (bool): If True, add a FileHandler for file output. Defaults to False.
    """
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
    """
    Rename a key according to a list of rules, using preset semantics first, then regex matching.

    Rules are processed in order. The first matching rule is applied and the function returns.
    Rule types include: "startswith", "endswith", "contains", "exact", "regex".

    Args:
        key (str): Original key name to process.
        rules (list): List of rules, each rule is a list of [type, pattern, replacement].
            type can be: "startswith", "endswith", "contains", "exact", "regex".

    Returns:
        renamed_key (str): Renamed key if a rule matches, otherwise returns the original key.
    """
    for rule in rules:
        if len(rule) != 3:
            print(f"Warning: Invalid rule format: {rule}")
            continue  # Skip invalid rule format

        rule_type, pattern, replacement = rule

        if rule_type == "startswith":
            if key.startswith(pattern):
                return replacement + key[len(pattern) :]

        elif rule_type == "endswith":
            if key.endswith(pattern):
                return key[: -len(pattern)] + replacement

        elif rule_type == "contains":
            if pattern in key:
                return key.replace(
                    pattern, replacement, 1
                )  # Replace only the first match

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

    return key  # Return original key if no match
