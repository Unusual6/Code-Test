"""
198. 打家劫舍
难度：中等
你是一个专业的小偷，计划偷窃沿街的房屋。相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
示例：输入：[1,2,3,1]  输出：4
"""
from typing import List

class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        prev, curr = nums[0], max(nums[0], nums[1])
        for i in range(2, len(nums)):
            prev, curr = curr, max(curr, prev + nums[i])
        return curr
