"""
101. 对称二叉树
难度：简单
给你一个二叉树的根节点 root，检查它是否轴对称。
示例：输入：root = [1,2,2,3,4,4,3]  输出：true
"""
from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        return self._isMirror(root, root)
    
    def _isMirror(self, t1, t2):
        if not t1 and not t2:
            return True
        if not t1 or not t2:
            return False
        return (t1.val == t2.val and 
                self._isMirror(t1.left, t2.right) and 
                self._isMirror(t1.right, t2.left))
