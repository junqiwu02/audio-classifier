# %%
import glob
import natsort
import ffmpeg
# %%
filelist = glob.glob('dataset/dev_splits_complete/*.mp4')
sr_no = 1
for f in natsort.os_sorted(filelist):
    stream = ffmpeg.input(f)
    out = ffmpeg.output(stream.audio, f'dataset/dev_splits_complete/wav/{sr_no}.wav')
    out.run()
    sr_no += 1


# %%
