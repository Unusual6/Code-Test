"""
35. 搜索插入位置
难度：简单
给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在，返回它将会被按顺序插入的位置。
示例：输入: nums = [1,3,5,6], target = 5  输出: 2
"""
from typing import List

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left
