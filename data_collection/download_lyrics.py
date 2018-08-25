import os

import pandas

from data_collection import lyrics_api


def save_lyrics(lyrics, outfile):
    with open(outfile, 'w') as file:
        file.write(lyrics)


def open_error_log():
    error_log = open('error_log', 'w')
    error_log.write("Errors during the download of the lyrics:\n")
    error_log.close()
    error_log = open('error_log', 'a')
    return error_log


if __name__ == '__main__':
    filename = 'data/lyrics_data.tsv'
    data = pandas.read_csv(filename, sep='\t', header=0)
    error_log = open_error_log()
    for idx, row in data.iterrows():
        artist = row[1]
        song = row[0]
        print('Downloading: %s - %s...' % (artist, song))
        (succeess, json) = lyrics_api.get_lyrics(artist.replace('/', ''), song.replace('/', ''))
        if succeess:
            lyrics = json['lyrics']
            outfile = os.path.join('data', 'raw_lyrics', artist + ' - ' + song)
            try:
                save_lyrics(lyrics, outfile)
            except Exception:
                error_log.write('Error saving data for %s - %s.\n' % (artist, song))
        else:
            error_log.write('Failed retrieval of lyrics for %s - %s.\n' % (artist, song))
    error_log.close()
