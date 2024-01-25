import logging
import os
import shutil


def rm_file_or_folder(path, is_file=None):
    if os.path.exists(path):
        try:
            if os.path.isdir(path) and not is_file:
                shutil.rmtree((path))
            else:
                os.unlink(path)
        except FileNotFoundError:
            logging.warning(f"Failed to remove '{path}' - file not found")
