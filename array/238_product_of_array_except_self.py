"""
238. 除自身以外数组的乘积 (Product of Array Except Self)
难度：中等
给你一个整数数组 nums，返回数组 answer，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积。
你必须在不使用除法的情况下在 O(n) 时间复杂度内完成此问题。
示例：输入：nums = [1,2,3,4]  输出：[24,12,8,6]
"""
from typing import List

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [1] * n
        left = 1
        for i in range(n):
            result[i] = left
            left *= nums[i]
        right = 1
        for i in range(n - 1, -1, -1):
            result[i] *= right
            right *= nums[i]
        return result
