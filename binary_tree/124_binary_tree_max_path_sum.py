"""
124. 二叉树中的最大路径和
难度：困难
二叉树中的路径被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。返回其最大路径和。
示例：输入：root = [1,2,3]  输出：6
"""
from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def __init__(self):
        self.max_sum = float('-inf')
    
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self._maxGain(root)
        return self.max_sum
    
    def _maxGain(self, node):
        if not node:
            return 0
        left_gain = max(self._maxGain(node.left), 0)
        right_gain = max(self._maxGain(node.right), 0)
        price_newpath = node.val + left_gain + right_gain
        self.max_sum = max(self.max_sum, price_newpath)
        return node.val + max(left_gain, right_gain)
