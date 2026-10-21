"""
230. 二叉搜索树中第K小的元素
难度：中等
给定一个二叉搜索树的根节点 root，和一个整数 k，查找其中第 k 个最小元素。
示例：输入：root = [3,1,4,null,2], k = 1  输出：1
"""
from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        current = root
        count = 0
        while current or stack:
            while current:
                stack.append(current)
                current = current.left
            current = stack.pop()
            count += 1
            if count == k:
                return current.val
            current = current.right
        return -1
