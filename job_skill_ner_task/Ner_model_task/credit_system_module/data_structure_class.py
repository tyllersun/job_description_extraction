class ListDict:
    def __init__(self):
        self.data = {}

    def convert_to_tuple(self, key):
        if isinstance(key, list):
            return tuple(key)
        return key

    def __getitem__(self, key):
        return self.data[self.convert_to_tuple(key)]

    def __setitem__(self, key, value):
        self.data[self.convert_to_tuple(key)] = value

    def __delitem__(self, key):
        del self.data[self.convert_to_tuple(key)]

    def __contains__(self, key):
        return self.convert_to_tuple(key) in self.data

    def keys(self):
        return [list(key) for key in self.data.keys()]

    def values(self):
        return self.data.values()

    def items(self):
        return [(list(key), value) for key, value in self.data.items()]

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return repr(self.data)


class credit_dict:
    def __init__(self, credit):
        self.dict_ = ListDict()
        self.credit = credit
        self.graduate = []

    def add_dict(self, candidate):
        if candidate in self.dict_:
            self.dict_[candidate] += 1
        else:
            self.dict_[candidate] = 1
        if self.dict_[candidate] >= self.credit:
            if candidate not in self.graduate:
                self.graduate.append(candidate)

    def get_keys(self):
        return self.dict_.keys()

    def get_graduate(self):
        return self.graduate
