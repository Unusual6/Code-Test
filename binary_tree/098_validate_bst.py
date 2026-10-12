"""
98. 验证二叉搜索树
难度：中等
给你一个二叉树的根节点 root，判断其是否是一个有效的二叉搜索树。
示例：输入：root = [2,1,3]  输出：true
"""
from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        return self._validate(root, float('-inf'), float('inf'))
    
    def _validate(self, node, low, high):
        if not node:
            return True
        if node.val <= low or node.val >= high:
            return False
        return (self._validate(node.left, low, node.val) and 
                self._validate(node.right, node.val, high))
