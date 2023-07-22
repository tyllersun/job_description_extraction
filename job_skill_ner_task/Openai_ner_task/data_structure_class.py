# Dict Tree

class TrieNode:
    def __init__(self):
        self.children = {}
        self.hasword = False # only True if exist word,
        # apple in dict, but app not in, False

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, list_of_word: list) -> None:
        cur = self.root
        for c in list_of_word:
            c = c.lower()
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.hasword = True

    def search(self, list_of_word: list) -> bool:
        cur  = self.root
        for c in list_of_word:
            c = c.lower()
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.hasword


    def startsWith(self, prefix_list: list) -> bool:
        cur = self.root
        for c in prefix_list:
            c = c.lower()
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return True


class ListDict:
    def __init__(self):
        self.data = {}

    def _convert_to_tuple(self, key):
        if isinstance(key, list):
            return tuple(key)
        return key

    def __getitem__(self, key):
        return self.data[self._convert_to_tuple(key)]

    def __setitem__(self, key, value):
        self.data[self._convert_to_tuple(key)] = value

    def __delitem__(self, key):
        del self.data[self._convert_to_tuple(key)]

    def __contains__(self, key):
        return self._convert_to_tuple(key) in self.data

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


class ListDict:
    def __init__(self):
        self.data = {}

    def _convert_to_tuple(self, key):
        if isinstance(key, list):
            return tuple(key)
        return key

    def __getitem__(self, key):
        return self.data[self._convert_to_tuple(key)]

    def __setitem__(self, key, value):
        self.data[self._convert_to_tuple(key)] = value

    def __delitem__(self, key):
        del self.data[self._convert_to_tuple(key)]

    def __contains__(self, key):
        return self._convert_to_tuple(key) in self.data

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

