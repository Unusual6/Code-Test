"""
300. 最长递增子序列
难度：中等
给你一个整数数组 nums，找到其中最长严格递增子序列的长度。
示例：输入：nums = [10,9,2,5,3,7,101,18]  输出：4
"""
from typing import List
import bisect

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        tails = []
        for num in nums:
            pos = bisect.bisect_left(tails, num)
            if pos == len(tails):
                tails.append(num)
            else:
                tails[pos] = num
        return len(tails)
