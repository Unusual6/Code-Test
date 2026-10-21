"""
33. 搜索旋转排序数组 (Search in Rotated Sorted Array)

难度：中等

题目描述：
整数数组 nums 按升序排列，数组中的值 互不相同 。
在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，
使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。
给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。
你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。

示例 1：
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4

示例 2：
输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1

示例 3：
输入：nums = [1], target = 0
输出：-1

提示：
1 <= nums.length <= 5000
-10^4 <= nums[i] <= 10^4
nums 中的每个值都 独一无二
题目数据保证 nums 在预先未知的某个下标上进行了旋转
-10^4 <= target <= 10^4
"""

from typing import List


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        """
        二分查找：判断哪一半是有序的，然后判断target是否在该范围内
        时间复杂度：O(log n)
        空间复杂度：O(1)
        """
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if nums[mid] == target:
                return mid
            
            # 判断左半部分是否有序
            if nums[left] <= nums[mid]:
                # 左半部分有序
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                # 右半部分有序
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return -1


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    nums1 = [4, 5, 6, 7, 0, 1, 2]
    target1 = 0
    result1 = solution.search(nums1, target1)
    print(f"输入: nums = {nums1}, target = {target1}")
    print(f"输出: {result1}")
    print()
    
    # 测试用例2
    nums2 = [4, 5, 6, 7, 0, 1, 2]
    target2 = 3
    result2 = solution.search(nums2, target2)
    print(f"输入: nums = {nums2}, target = {target2}")
    print(f"输出: {result2}")
