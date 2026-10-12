"""
287. 寻找重复数 (Find the Duplicate Number)
难度：中等
给定一个包含 n + 1 个整数的数组 nums，其数字都在 [1, n] 范围内（包括 1 和 n），可知至少存在一个重复的整数。
返回这个重复的数。
示例：输入：nums = [1,3,4,2,2]  输出：2
"""
from typing import List

class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # Floyd 判圈法
        slow = fast = nums[0]
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        slow = nums[0]
        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]
        return slow
