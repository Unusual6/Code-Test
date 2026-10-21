"""
236. 二叉树的最近公共祖先
难度：中等
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
示例：输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1  输出：3
"""

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def lowestCommonAncestor(self, root, p, q):
        if not root or root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        return left if left else right
