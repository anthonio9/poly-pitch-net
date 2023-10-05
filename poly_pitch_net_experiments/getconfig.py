import configparser
import json
import os
import sys
from pathlib import Path
from subprocess import call, check_output, STDOUT


def get_git_root():
    """
    Return None if p is not in a git repo,
    or the root of the repo if it is
    """
    if call(["git", "branch"], stderr=STDOUT,
            stdout=open(os.devnull, 'w')) != 0:
        return None
    else:
        root = check_output(["git", "rev-parse", "--show-toplevel"]).decode()

        # get rid of break `\n` symbol at the end of the path
        if root[-1:] == "\n":
            root = root[:-1]
        root = Path(root)
        return root


def get_datasets_path():
    datasets_path_json = config['CUSTOM_PATH']['datasets_path']
    dataset_path_list = json.loads(datasets_path_json)

    return os.path.join(*dataset_path_list)


def get_models_path():
    models_path_json = config['CUSTOM_PATH']['models_path']
    models_path_list = json.loads(models_path_json)

    return os.path.join(*models_path_list)


# add git root path to the sys.path
git_root_path = get_git_root()

if git_root_path is not None:

    print(f"git_root_path: {git_root_path}")
    sys.path.append(str(git_root_path))

# parse paths from the config file
config = configparser.ConfigParser()
config.read('config.ini')
