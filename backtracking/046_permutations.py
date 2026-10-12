"""
46. 全排列
难度：中等
给定一个不含重复数字的数组 nums，返回其所有可能的全排列。
示例：输入：nums = [1,2,3]  输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
"""
from typing import List

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result = []
        self._backtrack(nums, [], set(), result)
        return result
    
    def _backtrack(self, nums, path, used, result):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for num in nums:
            if num in used:
                continue
            used.add(num)
            path.append(num)
            self._backtrack(nums, path, used, result)
            path.pop()
            used.remove(num)
