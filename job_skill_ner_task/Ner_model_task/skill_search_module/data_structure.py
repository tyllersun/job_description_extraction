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

