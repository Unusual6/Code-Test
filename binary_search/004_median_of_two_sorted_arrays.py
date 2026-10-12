"""
4. 寻找两个正序数组的中位数
难度：困难
给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的中位数。
示例：输入：nums1 = [1,3], nums2 = [2]  输出：2.00000
"""
from typing import List

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        m, n = len(nums1), len(nums2)
        left, right = 0, m
        while left <= right:
            i = (left + right) // 2
            j = (m + n + 1) // 2 - i
            max_left1 = float('-inf') if i == 0 else nums1[i-1]
            min_right1 = float('inf') if i == m else nums1[i]
            max_left2 = float('-inf') if j == 0 else nums2[j-1]
            min_right2 = float('inf') if j == n else nums2[j]
            if max_left1 <= min_right2 and max_left2 <= min_right1:
                if (m + n) % 2 == 1:
                    return max(max_left1, max_left2)
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
            elif max_left1 > min_right2:
                right = i - 1
            else:
                left = i + 1
        return 0.0
