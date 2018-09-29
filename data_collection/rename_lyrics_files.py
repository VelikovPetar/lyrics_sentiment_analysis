import os

from common_utils import get_annotated_dataset, get_full_dataset

if __name__ == '__main__':
    full_dataset = get_full_dataset()
    annotated_dataset = get_annotated_dataset()

    raw_lyrics_dir = '../data/raw_lyrics'

    for _, row in full_dataset.iterrows():
        artist = row['Artist']
        song_name = row[0]
        song_code = row['Code']
        full_name = artist + ' - ' + song_name
        full_name = raw_lyrics_dir + '/' + full_name
        if os.path.exists(full_name):
            os.rename(full_name, raw_lyrics_dir + '/' + song_code + '.txt')
        else:
            print("Can't rename %s" % full_name)
