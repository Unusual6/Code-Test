"""
78. 子集
难度：中等
给你一个整数数组 nums，数组中的元素互不相同。返回该数组所有可能的子集。
示例：输入：nums = [1,2,3]  输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
"""
from typing import List

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        result = []
        self._backtrack(nums, 0, [], result)
        return result
    
    def _backtrack(self, nums, start, path, result):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            self._backtrack(nums, i + 1, path, result)
            path.pop()
