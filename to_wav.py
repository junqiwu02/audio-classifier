import sys
import os

import glob
import natsort
from tqdm import tqdm

import ffmpeg


if __name__ == '__main__':
    filelist = natsort.os_sorted(glob.glob(f'./data/{sys.argv[1]}/*.mp4'))
    for f in tqdm(filelist):
        basename = os.path.basename(f)
        basename = os.path.splitext(basename)[0]
        try:
            stream = ffmpeg.input(f)
            out = ffmpeg.output(stream.audio, f'./data/{sys.argv[1]}/wav/{basename}.wav', loglevel='quiet')
            out.run()
        except:
            print(f'Failed to convert {basename}, continuing...')

