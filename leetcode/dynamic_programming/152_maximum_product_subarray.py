"""
152. 乘积最大子数组
难度：中等
给你一个整数数组 nums，找出数组中乘积最大的非空连续子数组，并返回该子数组所对应的乘积。
示例：输入: nums = [2,3,-2,4]  输出: 6
"""
from typing import List

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max_prod = min_prod = result = nums[0]
        for i in range(1, len(nums)):
            if nums[i] < 0:
                max_prod, min_prod = min_prod, max_prod
            max_prod = max(nums[i], max_prod * nums[i])
            min_prod = min(nums[i], min_prod * nums[i])
            result = max(result, max_prod)
        return result
