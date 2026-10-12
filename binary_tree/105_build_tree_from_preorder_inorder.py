"""
105. 从前序与中序遍历序列构造二叉树
难度：中等
给定两个整数数组 preorder 和 inorder，构造二叉树并返回其根节点。
示例：输入：preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]  输出：[3,9,20,null,null,15,7]
"""
from typing import Optional, List

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1:mid + 1], inorder[:mid])
        root.right = self.buildTree(preorder[mid + 1:], inorder[mid + 1:])
        return root
