import dfm

if __name__ == '__main__':
    # dfm.vid2imgs("source_data/", "source_video.mp4", 366, 367, skip_frames=True)
    # dfm.cleaner("source_data/")
    # dfm.create_data("source_data/")
    dfm.vid2imgs("target_data/", "target_video.mp4", 50, 60)
    dfm.cleaner("target_data/")
    dfm.create_data("target_data/")
    dfm.best_swapper("source_data/", "target_data/", "processed_data/")
    dfm.vid_mod("target_data/", "processed_data/", "target_video.mp4", "modded.mp4")
