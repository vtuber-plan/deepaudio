

import json

class HParams(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v
    
    @staticmethod
    def from_json_file(path: str):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.loads(f.read())
        hparams = HParams(**obj)
        return hparams

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()
    
    def get(self, k, v=None):
        if k not in self.keys():
            return v
        else:
            return self.__getitem__(k)

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
