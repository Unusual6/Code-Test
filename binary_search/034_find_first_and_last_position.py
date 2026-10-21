"""
34. 在排序数组中查找元素的第一个和最后一个位置 (Find First and Last Position of Element in Sorted Array)

难度：中等

题目描述：
给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。
如果数组中不存在目标值 target，返回 [-1, -1] 。
你必须设计并实现时间复杂度为 O(log n) 的算法解决此问题。

示例 1：
输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]

示例 2：
输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]

示例 3：
输入：nums = [], target = 0
输出：[-1,-1]

提示：
0 <= nums.length <= 10^5
-10^9 <= nums[i] <= 10^9
nums 是一个非递减数组
-10^9 <= target <= 10^9
"""

from typing import List


class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        """
        两次二分查找：分别找第一个和最后一个位置
        时间复杂度：O(log n)
        空间复杂度：O(1)
        """
        def findFirst(nums, target):
            left, right = 0, len(nums) - 1
            first = -1
            
            while left <= right:
                mid = (left + right) // 2
                
                if nums[mid] == target:
                    first = mid
                    right = mid - 1  # 继续向左找
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return first
        
        def findLast(nums, target):
            left, right = 0, len(nums) - 1
            last = -1
            
            while left <= right:
                mid = (left + right) // 2
                
                if nums[mid] == target:
                    last = mid
                    left = mid + 1  # 继续向右找
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return last
        
        if not nums:
            return [-1, -1]
        
        first = findFirst(nums, target)
        if first == -1:
            return [-1, -1]
        
        last = findLast(nums, target)
        return [first, last]


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    nums1 = [5, 7, 7, 8, 8, 10]
    target1 = 8
    result1 = solution.searchRange(nums1, target1)
    print(f"输入: nums = {nums1}, target = {target1}")
    print(f"输出: {result1}")
    print()
    
    # 测试用例2
    nums2 = [5, 7, 7, 8, 8, 10]
    target2 = 6
    result2 = solution.searchRange(nums2, target2)
    print(f"输入: nums = {nums2}, target = {target2}")
    print(f"输出: {result2}")
