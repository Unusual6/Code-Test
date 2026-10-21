"""
94. 二叉树的中序遍历
难度：简单
给定一个二叉树的根节点 root，返回它的中序遍历。
示例：输入：root = [1,null,2,3]  输出：[1,3,2]
"""
from typing import Optional, List

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        result = []
        self._inorder(root, result)
        return result
    
    def _inorder(self, node, result):
        if not node:
            return
        self._inorder(node.left, result)
        result.append(node.val)
        self._inorder(node.right, result)
