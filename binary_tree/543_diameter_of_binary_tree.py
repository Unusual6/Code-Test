"""
543. 二叉树的直径
难度：简单
给你一棵二叉树的根节点，返回该树的直径。
示例：输入：root = [1,2,3,4,5]  输出：3
"""
from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def __init__(self):
        self.max_diameter = 0
    
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self._depth(root)
        return self.max_diameter
    
    def _depth(self, node):
        if not node:
            return 0
        left = self._depth(node.left)
        right = self._depth(node.right)
        self.max_diameter = max(self.max_diameter, left + right)
        return 1 + max(left, right)
