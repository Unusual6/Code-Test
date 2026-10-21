"""
560. 和为 K 的子数组 (Subarray Sum Equals K)
难度：中等
给你一个整数数组 nums 和一个整数 k，请你统计并返回该数组中和为 k 的连续子数组的个数。
示例：输入：nums = [1,1,1], k = 2  输出：2
"""
from typing import List
from collections import defaultdict

class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        prefix_sum_count = defaultdict(int)
        prefix_sum_count[0] = 1
        current_sum = 0
        count = 0
        for num in nums:
            current_sum += num
            count += prefix_sum_count[current_sum - k]
            prefix_sum_count[current_sum] += 1
        return count
