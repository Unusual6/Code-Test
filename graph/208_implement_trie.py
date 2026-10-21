"""
208. 实现 Trie (前缀树)
难度：中等
实现 Trie 类，支持 insert, search, startsWith 操作。
"""

class Trie:
    def __init__(self):
        self.children = {}
        self.is_end = False
    
    def insert(self, word: str) -> None:
        node = self
        for char in word:
            if char not in node.children:
                node.children[char] = Trie()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word: str) -> bool:
        node = self._find(word)
        return node is not None and node.is_end
    
    def startsWith(self, prefix: str) -> bool:
        return self._find(prefix) is not None
    
    def _find(self, prefix):
        node = self
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
