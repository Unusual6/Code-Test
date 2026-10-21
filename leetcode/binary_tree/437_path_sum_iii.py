"""
437. 路径总和 III
难度：中等
求该二叉树里节点值之和等于 targetSum 的路径的数目。
示例：输入：root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8  输出：3
"""
from typing import Optional
from collections import defaultdict

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        prefix_sum_count = defaultdict(int)
        prefix_sum_count[0] = 1
        return self._dfs(root, targetSum, 0, prefix_sum_count)
    
    def _dfs(self, node, targetSum, current_sum, prefix_sum_count):
        if not node:
            return 0
        current_sum += node.val
        count = prefix_sum_count[current_sum - targetSum]
        prefix_sum_count[current_sum] += 1
        count += self._dfs(node.left, targetSum, current_sum, prefix_sum_count)
        count += self._dfs(node.right, targetSum, current_sum, prefix_sum_count)
        prefix_sum_count[current_sum] -= 1
        return count
