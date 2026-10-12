"""
114. 二叉树展开为链表
难度：中等
给你二叉树的根结点 root，将它展开为一个单链表。
示例：输入：root = [1,2,5,3,4,null,6]  输出：[1,null,2,null,3,null,4,null,5,null,6]
"""
from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        if not root:
            return
        self.flatten(root.left)
        self.flatten(root.right)
        left = root.left
        right = root.right
        root.left = None
        root.right = left
        curr = root
        while curr.right:
            curr = curr.right
        curr.right = right
