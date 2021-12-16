from collect_data import create_data_folder, collect_train_data
from train_model import train

if __name__ == "__main__":
    class_label = ['NA', 'AN', 'DI', 'FE', 'HA', 'SA', 'SU']

    # Step 1. Automatic create train/val folders
    create_data_folder()

    # Step 2. Data collection
    # - Collect train/val data by setting 'exp' (expression) with: natural / angry / disgust / fear / happy / sad / surprise
    collect_train_data(exp="surprise", save=True)

    # Step 3. Network training
    train()

