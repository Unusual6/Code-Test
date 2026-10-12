"""
104. 二叉树的最大深度
难度：简单
给定一个二叉树 root，返回其最大深度。
示例：输入：root = [3,9,20,null,null,15,7]  输出：3
"""
from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
