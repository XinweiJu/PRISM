from __future__ import absolute_import, division, print_function

import os


def env_path(env_name, default_root_env, default_root, *parts):
    """Resolve a path from a specific env var or a generic project root."""
    if os.environ.get(env_name):
        return os.environ[env_name]
    root = os.environ.get(default_root_env, default_root)
    return os.path.join(root, *parts)


def data_path(*parts):
    return env_path("PRISM_DATA_PATH", "PRISM_DATA_ROOT", os.path.join("data_folder", "input_data"), *parts)


def generated_path(*parts):
    return env_path("PRISM_GENERATED_PATH", "PRISM_DATA_ROOT", os.path.join("data_folder", "generated"), *parts)


def weights_path(*parts):
    return env_path("PRISM_WEIGHTS_PATH", "PRISM_WEIGHTS_ROOT", "weights", *parts)


def output_path(*parts):
    return env_path("PRISM_OUTPUT_PATH", "PRISM_OUTPUT_ROOT", "output_data", *parts)
