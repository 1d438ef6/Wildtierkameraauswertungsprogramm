from typing import Dict, List
from dataclasses import dataclass
from collections import Counter


class Node:
    char: chr

    def __init__(self, char: chr):
        self.char = char

    def __str__(self):
        return self.char


class Layer:

    nodes: List[Node]

    def contains(self, item):
        for node in self.nodes:
            if item == node.char:
                return True
        return False

    def __init__(self):
        self.nodes = []

    def __call__(self, char: chr, *args, **kwargs):
        if not self.contains(char):
            self.nodes.append(Node(char))

    def __str__(self):
        return str([node.char for node in self.nodes])

    def avg(self):
        l = [node.char for node in self.nodes]
        return max(l, key=Counter(l).get)

class Tree:

    _word: str = ""
    _layers = List[Layer]
    _n_layer: int

    def __init__(self, word: str, n_layer:int=None):
        self.word = word
        self._n_layer = n_layer if n_layer else len(word)
        self._layers = []

        self.add(word)

    def add(self, word: str):
        while len(word)<self.n_layer:
            word += " "
        for num in range(len(word)):
            if len(self.layers) < num + 1:
                self.layers.append(Layer())

            self.layers[num](word[num])

    @property
    def layers(self):
        return self._layers

    @property
    def word(self):
        return

    @word.setter
    def word(self, value):
        pass

    @property
    def n_layer(self):
        return self._n_layer


if __name__ == "__main__":
    def create_tree(content: str):

        tree = Tree(content,n_layer=9)
        tree.add("DWR16421")
        tree.add("DWR1642F")
        tree.add("LDER16426")

        print(tree)
        print(tree.n_layer)

        for layer in tree.layers:
            print(layer.avg())

    create_tree("DWR1642F")