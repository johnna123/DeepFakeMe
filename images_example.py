import dfm

if __name__ == '__main__':
    dfm.cleaner("source_data/")
    dfm.cleaner("target_data/")

    dfm.create_data("source_data/")
    dfm.create_data("target_data/")

    dfm.best_swapper("source_data/", "target_data/", "processed_data/")
