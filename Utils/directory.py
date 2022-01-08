import os
import errno

def make_dir(file):

    try:

        os.mkdir(file)

    except OSError as e:

        if e.errno != errno.EEXIST:

            raise