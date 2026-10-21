"""
88. 合并两个有序数组 (Merge Sorted Array)

难度：简单

题目描述：
给你两个按 非递减顺序 排列的整数数组 nums1 和 nums2，另有两个整数 m 和 n ，分别表示 nums1 和 nums2 中的元素数目。
请你 合并 nums2 到 nums1 中，使合并后的数组同样按 非递减顺序 排列。
注意：最终，合并后数组不应由函数返回，而是存储在数组 nums1 中。
为了应对这种情况，nums1 的初始长度为 m + n，其中前 m 个元素表示应合并的元素，后 n 个元素为 0 ，应忽略。
nums2 的长度为 n 。

示例 1：
输入：nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
输出：[1,2,2,3,5,6]

示例 2：
输入：nums1 = [1], m = 1, nums2 = [], n = 0
输出：[1]

示例 3：
输入：nums1 = [0], m = 0, nums2 = [1], n = 1
输出：[1]

提示：
nums1.length == m + n
nums2.length == n
0 <= m, n <= 200
1 <= m + n <= 200
nums1[i] 和 nums2[i] 均为 -10^9 到 10^9 之间的整数

进阶：你可以设计实现一个时间复杂度为 O(m + n) 的算法解决此问题吗？
"""

from typing import List


class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        从后往前合并，避免覆盖 nums1 中的元素
        时间复杂度：O(m + n)
        空间复杂度：O(1)
        Do not return anything, modify nums1 in-place instead.
        """
        p1 = m - 1  # nums1 有效元素的最后一个
        p2 = n - 1  # nums2 的最后一个
        p = m + n - 1  # 合并后数组的最后一个位置
        
        # 从后往前比较并放置
        while p1 >= 0 and p2 >= 0:
            if nums1[p1] > nums2[p2]:
                nums1[p] = nums1[p1]
                p1 -= 1
            else:
                nums1[p] = nums2[p2]
                p2 -= 1
            p -= 1
        
        # 如果 nums2 还有剩余，复制到 nums1
        while p2 >= 0:
            nums1[p] = nums2[p2]
            p2 -= 1
            p -= 1


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    nums1_1 = [1, 2, 3, 0, 0, 0]
    m1 = 3
    nums2_1 = [2, 5, 6]
    n1 = 3
    print(f"输入: nums1 = {nums1_1}, m = {m1}, nums2 = {nums2_1}, n = {n1}")
    solution.merge(nums1_1, m1, nums2_1, n1)
    print(f"输出: {nums1_1}")
    print()
    
    # 测试用例2
    nums1_2 = [1]
    m2 = 1
    nums2_2 = []
    n2 = 0
    print(f"输入: nums1 = {nums1_2}, m = {m2}, nums2 = {nums2_2}, n = {n2}")
    solution.merge(nums1_2, m2, nums2_2, n2)
    print(f"输出: {nums1_2}")
