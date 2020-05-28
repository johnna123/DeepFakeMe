import dfm

if __name__ == '__main__':
    dfm.cleaner("source_data/")
    dfm.create_data("source_data/")
    dfm.vid2img("target_data/", "target_video.mp4", 110, 125)
    dfm.cleaner("target_data/")
    dfm.create_data("target_data/")
    dfm.best_swapper("source_data/", "target_data/", "processed_data/")
    # dfm.vid_mod("target_data/", "processed_data/", "target_video.mp4", "modded.mp4")
