"""
199. 二叉树的右视图
难度：中等
给定一个二叉树的根节点 root，想象自己站在它的右侧，返回从右侧所能看到的节点值。
示例：输入：root = [1,2,3,null,5,null,4]  输出：[1,3,4]
"""
from typing import Optional, List
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        result = []
        queue = deque([root])
        while queue:
            level_size = len(queue)
            for i in range(level_size):
                node = queue.popleft()
                if i == level_size - 1:
                    result.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return result
