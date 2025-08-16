

import kagglehub
import shutil
import os

def download_musicnet():
    try:
        path = kagglehub.dataset_download("imsparsh/musicnet-dataset")
        print("Dataset downloaded successfully to:", path)

        target_folder = os.path.join(os.getcwd(), 'musicnet_dataset')

        if not os.path.exists(target_folder):
            shutil.copytree(path, target_folder)
            print("Dataset copied to:", target_folder)
        else:
            print("Target folder already exists, skipping copy.")

    except Exception as e:
        print("Failed to download via kagglehub:", e)
        print("Please manually upload 'musicnet-dataset.zip' and create and export into 'musicnet_dataset' folder")

if __name__ == "__main__":
    download_musicnet()
