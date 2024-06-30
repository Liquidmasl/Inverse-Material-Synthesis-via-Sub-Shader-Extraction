


import dropbox
from dropbox.exceptions import AuthError
import concurrent.futures


def download_file(file):
    if isinstance(file, dropbox.files.FileMetadata):
        local_path = f"{local_directory}/{file.name}"

        try:
            dbx.files_download_to_file(local_path, file.path_lower)
            print(f"Downloaded {file.path_lower} to {local_path}")
        except dropbox.exceptions.ApiError as err:
            print(f"Failed to download {file.path_lower}: {err}")

# Replace with your Dropbox API access token
ACCESS_TOKEN = 'sl.B3y72sdv5N-Ehk4GjmmttSOqraCL7Lzryt5MoYBAJ2M_pKl0RclNi6Xu9J7QpL9ns_xjYOjI1PpnF1me_vXRR3qegMGqU7Yn4Lyj6Kw5beRm18LyizmOv1_dKh8U8NeL4QSCg-TFyOvXY71MZi1PCX8'

# Initialize a Dropbox object
dbx = dropbox.Dropbox(ACCESS_TOKEN)

# Verify that the access token is valid
try:
    dbx.users_get_current_account()
except AuthError:
    print("ERROR: Invalid access token; try re-generating an access token from the app console on the web.")

# The shared folder path
shared_folder_path = r'/Masl Stuff/BachelorArbeit/Datasets/BiggerShaderDataset/renders'

# The local directory where you want to download the files
local_directory = '/app/data/renders'

# List all files in the shared folder
try:
    response = dbx.files_list_folder(shared_folder_path)
    files = response.entries
except dropbox.exceptions.ApiError as err:
    print(f"Folder listing failed for {shared_folder_path} -- assumed empty: {err}")
    files = []


while True:
    # Download each file
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(download_file, files)


    # Check if there are more files in the shared folder
    if not response.has_more:
        break

    # Get the next set of files
    response = dbx.files_list_folder_continue(response.cursor)
    files = response.entries