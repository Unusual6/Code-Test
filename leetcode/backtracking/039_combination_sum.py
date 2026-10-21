"""
39. 组合总和
难度：中等
给你一个无重复元素的整数数组 candidates 和一个目标整数 target，找出 candidates 中可以使数字和为目标数 target 的所有不同组合。
示例：输入：candidates = [2,3,6,7], target = 7  输出：[[2,2,3],[7]]
"""
from typing import List

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        result = []
        self._backtrack(candidates, target, 0, [], result)
        return result
    
    def _backtrack(self, candidates, target, start, path, result):
        if target == 0:
            result.append(path[:])
            return
        if target < 0:
            return
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            self._backtrack(candidates, target - candidates[i], i, path, result)
            path.pop()
