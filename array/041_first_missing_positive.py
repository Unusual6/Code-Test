"""
41. 缺失的第一个正数 (First Missing Positive)

难度：困难

题目描述：
给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。
请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。

示例 1：
输入：nums = [1,2,0]
输出：3

示例 2：
输入：nums = [3,4,-1,1]
输出：2

示例 3：
输入：nums = [7,8,9,11,12]
输出：1

提示：
1 <= nums.length <= 10^5
-2^31 <= nums[i] <= 2^31 - 1
"""

from typing import List


class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        """
        原地哈希：将数字 i 放到位置 i-1
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        n = len(nums)
        
        # 将每个数放到正确的位置
        for i in range(n):
            while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
                # 交换 nums[i] 和 nums[nums[i] - 1]
                correct_pos = nums[i] - 1
                nums[i], nums[correct_pos] = nums[correct_pos], nums[i]
        
        # 找到第一个不在正确位置的数
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
        
        # 如果所有位置都正确，答案是 n + 1
        return n + 1


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    nums1 = [1, 2, 0]
    result1 = solution.firstMissingPositive(nums1)
    print(f"输入: nums = {nums1}")
    print(f"输出: {result1}")
    print()
    
    # 测试用例2
    nums2 = [3, 4, -1, 1]
    result2 = solution.firstMissingPositive(nums2)
    print(f"输入: nums = {nums2}")
    print(f"输出: {result2}")
    print()
    
    # 测试用例3
    nums3 = [7, 8, 9, 11, 12]
    result3 = solution.firstMissingPositive(nums3)
    print(f"输入: nums = {nums3}")
    print(f"输出: {result3}")
