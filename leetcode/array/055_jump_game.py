"""
55. 跳跃游戏 (Jump Game)
难度：中等
给你一个非负整数数组 nums，你最初位于数组的第一个下标。数组中的每个元素代表你在该位置可以跳跃的最大长度。
判断你是否能够到达最后一个下标。
示例：输入：nums = [2,3,1,1,4]  输出：true
"""
from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_reach = 0
        for i in range(len(nums)):
            if i > max_reach:
                return False
            max_reach = max(max_reach, i + nums[i])
            if max_reach >= len(nums) - 1:
                return True
        return True
