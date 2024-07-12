from random import randint
import yaml
import imp
import importlib.machinery
import importlib.util
from pathlib import Path, PurePath
import os
import subprocess
import sys
from random import randint


def plugin_type_key():
    return "type"


def plugin_name_key():
    return "name"


def plugin_description_key():
    return "description"


def plugin_version_key():
    return "version"


def plugin_file_key():
    return "file"


def plugin_factory_class_name_key():
    return "factory_class_name"


def load_plugin_yaml(filepath) -> dict:
    plugin_info = dict()
    with open(filepath, "r") as f:
        yaml_content = yaml.full_load(f)
        try:
            plugin_info[plugin_type_key()] = yaml_content[plugin_type_key()]
            plugin_info[plugin_name_key()] = yaml_content[plugin_name_key()]
            plugin_info[plugin_description_key()] = yaml_content[
                plugin_description_key()
            ]
            plugin_info[plugin_version_key()] = yaml_content[plugin_version_key()]

            # Todo: input / output
        except Exception as err:
            raise KeyError(f"Error loading plugin file {filepath}, see {err}")

    return plugin_info


def install_requirements(folderpath):
    filepath = os.path.abspath(os.path.join(folderpath, "requirements.txt"))

    if Path(filepath).is_file():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=folderpath,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )


def import_plugin_module_old(folderpath, factory_super_class_name):
    folderpath = os.path.abspath(folderpath)
    module_name = "main.py"
    factory_class_name = "PluginFactory"

    filepath = os.path.join(folderpath, module_name)

    dir_name = os.path.dirname(filepath)
    if dir_name not in sys.path:
        sys.path.append(dir_name)

    loader = importlib.machinery.SourceFileLoader(module_name, filepath)
    spec = importlib.util.spec_from_loader(module_name, loader)
    plugin_module = importlib.util.module_from_spec(spec)
    loader.exec_module(plugin_module)

    assert hasattr(
        plugin_module, factory_class_name
    ), f"class {factory_class_name} is not in module {module_name}"
    plugin_class = getattr(plugin_module, factory_class_name)

    has_correct_superclass = False
    for parent in plugin_class.__bases__:
        parent_name = parent.__name__
        if parent_name == factory_super_class_name:
            has_correct_superclass = True
    assert (
        has_correct_superclass
    ), f"class {factory_class_name} should inherit from {factory_super_class_name}"

    return plugin_class


def import_plugin_module(folderpath, factory_super_class_name):
    folderpath = Path(folderpath).absolute()
    factory_class_name = "PluginFactory"
    filepath = folderpath / "main.py"

    name = PurePath(Path(filepath)).parent.name
    install_requirements(folderpath=folderpath)
    module_name = str(Path(folderpath)).replace("\\", ".")
    spec = importlib.util.spec_from_file_location(name, filepath)
    plugin_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin_module)
    sys.modules[name + ".main"] = plugin_module

    assert hasattr(
        plugin_module, factory_class_name
    ), f"class {factory_class_name} is not in module {module_name}"
    plugin_class = getattr(plugin_module, factory_class_name)

    has_correct_superclass = False
    for parent in plugin_class.__bases__:
        parent_name = parent.__name__
        if parent_name == factory_super_class_name:
            has_correct_superclass = True
    assert (
        has_correct_superclass
    ), f"class {factory_class_name} should inherit from {factory_super_class_name}"

    return plugin_class
