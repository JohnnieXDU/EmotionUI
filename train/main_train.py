from collect_data import create_data_folder
from train_model import train

if __name__ == "__main__":
    class_label = ['NA', 'AN', 'DI', 'FE', 'HA', 'SA', 'SU']

    # Step 1. Collect data
    create_data_folder()

    # classes: natural / angry / disgust / fear / happy / sad / surprise
    # collect_train_data(exp="surprise", save=True)

    # Step 2. Train
    train()

