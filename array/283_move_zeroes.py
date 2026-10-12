"""
283. 移动零 (Move Zeroes)

难度：简单

题目描述：
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
请注意 ，必须在不复制数组的情况下原地对数组进行操作。

示例 1:
输入: nums = [0,1,0,3,12]
输出: [1,3,12,0,0]

示例 2:
输入: nums = [0]
输出: [0]

提示:
1 <= nums.length <= 10^4
-2^31 <= nums[i] <= 2^31 - 1

进阶：你能尽量减少完成的操作次数吗？
"""

from typing import List


class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        双指针法：左指针指向下一个非零元素应该放置的位置，右指针遍历数组
        时间复杂度：O(n)
        空间复杂度：O(1)
        Do not return anything, modify nums in-place instead.
        """
        left = 0  # 指向下一个非零元素应该放置的位置
        
        for right in range(len(nums)):
            if nums[right] != 0:
                # 交换左右指针指向的元素
                nums[left], nums[right] = nums[right], nums[left]
                left += 1


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    nums1 = [0, 1, 0, 3, 12]
    print(f"输入: nums = {nums1}")
    solution.moveZeroes(nums1)
    print(f"输出: {nums1}")
    print()
    
    # 测试用例2
    nums2 = [0]
    print(f"输入: nums = {nums2}")
    solution.moveZeroes(nums2)
    print(f"输出: {nums2}")
