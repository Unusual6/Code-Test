"""
572. 另一棵树的子树
难度：简单
给你两棵二叉树 root 和 subRoot，检验 root 中是否包含和 subRoot 具有相同结构和相同节点值的子树。
示例：输入：root = [3,4,5,1,2], subRoot = [4,1,2]  输出：true
"""
from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if not root:
            return False
        if self._isSameTree(root, subRoot):
            return True
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
    
    def _isSameTree(self, p, q):
        if not p and not q:
            return True
        if not p or not q:
            return False
        return (p.val == q.val and 
                self._isSameTree(p.left, q.left) and 
                self._isSameTree(p.right, q.right))
