from dataclasses import dataclass, field
from typing import List


@dataclass
class PrefixTreeNode:
    children: dict = field(default_factory=dict)
    is_word: bool = False


class PrefixTree:
    def __init__(self, words: List[str]):
        self.root = PrefixTreeNode()
        self._add_words(words)

    def _add_word(self, text: str):
        node = self.root
        for i in range(len(text)):
            c = text[i]  # current char
            if c not in node.children:
                node.children[c] = PrefixTreeNode()
            node = node.children[c]
            is_last = i + 1 == len(text)
            if is_last:
                node.is_word = True

    def _add_words(self, words: List[str]):
        for w in words:
            self._add_word(w)

    def _get_node(self, text: str):
        node = self.root
        for c in text:
            if c in node.children:
                node = node.children[c]
            else:
                return None
        return node

    def is_word(self, text: str) -> bool:
        node = self._get_node(text)
        if node:
            return node.is_word
        return False

    def get_next_chars(self, text: str) -> List[str]:
        chars = []
        node = self._get_node(text)
        if node:
            for k in node.children.keys():
                chars.append(k)
        return chars
