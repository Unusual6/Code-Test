"""
239. 滑动窗口最大值 (Sliding Window Maximum)
难度：困难
给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到最右侧。
你只能看到在滑动窗口内的 k 个数字。返回滑动窗口中的最大值。
示例：输入：nums = [1,3,-1,-3,5,3,6,7], k = 3  输出：[3,3,5,5,6,7]
"""
from typing import List
from collections import deque

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if not nums:
            return []
        result = []
        window = deque()
        for i in range(len(nums)):
            while window and window[0] < i - k + 1:
                window.popleft()
            while window and nums[window[-1]] < nums[i]:
                window.pop()
            window.append(i)
            if i >= k - 1:
                result.append(nums[window[0]])
        return result
