from os.path import dirname, realpath, join

class Config:
    base_dir = realpath(join(dirname(realpath(__file__)), '..'))
    data_dir = join(base_dir, "data")
