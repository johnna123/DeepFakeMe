import os


def delete_data(data_dir):
    files = os.listdir(data_dir)
    for file in files:
        if ".mp4" not in file:
            os.remove(data_dir+file)


if __name__ == '__main__':
    delete_data("source_data/")
    delete_data("target_data/")
    delete_data("processed_data/")
