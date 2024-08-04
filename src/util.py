import os
import re


def get_file_with_highest_number(folder_path, name_pattern, return_num=False):
    """
    Returns the path to the file in the specified folder that matches the name pattern
    with the highest running number.

    Args:
        folder_path (str): The path to the folder containing the files.
        name_pattern (str): The name pattern to match files.

    Returns:
        str: The path to the file with the highest running number.
    """
    pattern = re.compile(name_pattern.replace("##", r"(\d+)"))

    highest_number = -1
    highest_file = None

    # List all files in the folder
    for file_name in os.listdir(folder_path):
        match = pattern.match(file_name)
        if match:
            # Extract the running number
            number = int(match.group(1))
            if number > highest_number:
                highest_number = number
                highest_file = file_name



    if highest_file:
        path = os.path.join(folder_path, highest_file)

        if return_num:
            return path, highest_number
        return path
    else:
        return None