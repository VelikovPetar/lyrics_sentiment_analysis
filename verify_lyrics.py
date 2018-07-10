import os

import pandas

if __name__ == '__main__':
    filename = 'data/lyrics_data.tsv'
    data = pandas.read_csv(filename, sep='\t', header=0)
    mismatches = 0
    for idx, row in data.iterrows():
        artist = row[1]
        song = row[0]
        filename = '%s - %s' % (artist, song)
        filename = os.path.join('data', 'raw_lyrics', filename)
        if not os.path.exists(filename):
            print(filename)
            mismatches += 1
    print('Total mismatches: %d/%d' % (mismatches, data.shape[0]))
