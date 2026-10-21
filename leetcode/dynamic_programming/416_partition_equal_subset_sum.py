"""
416. 分割等和子集
难度：中等
给你一个只包含正整数的非空数组 nums。判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
示例：输入：nums = [1,5,11,5]  输出：true
"""
from typing import List

class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2 != 0:
            return False
        target = total // 2
        dp = [False] * (target + 1)
        dp[0] = True
        for num in nums:
            for j in range(target, num - 1, -1):
                dp[j] = dp[j] or dp[j - num]
        return dp[target]
