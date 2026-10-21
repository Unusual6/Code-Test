"""
162. 寻找峰值 (Find Peak Element)

难度：中等

题目描述：
峰值元素是指其值严格大于左右相邻值的元素。
给你一个整数数组 nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 即可。
你可以假设 nums[i] != -1 对于所有有效的 i 。
你可以想象成 nums[-1] = nums[n] = -∞ 。
你必须实现时间复杂度为 O(log n) 的算法来解决此问题。

示例 1：
输入：nums = [1,2,3,1]
输出：2
解释：3 是峰值元素，你的函数应该返回其索引 2。

示例 2：
输入：nums = [1,2,1,3,5,6,4]
输出：1 或 5 
解释：你的函数可以返回索引 1，其峰值元素为 2；或者返回索引 5， 其峰值元素为 6。

提示：
1 <= nums.length <= 1000
-2^31 <= nums[i] <= 2^31 - 1
对于所有有效的 i 都有 nums[i] != nums[i + 1]
"""

from typing import List


class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        """
        二分查找：如果中间元素小于右边元素，峰值在右边；否则在左边
        时间复杂度：O(log n)
        空间复杂度：O(1)
        """
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = (left + right) // 2
            
            # 如果中间元素小于右边元素，峰值在右边
            if nums[mid] < nums[mid + 1]:
                left = mid + 1
            else:
                # 否则峰值在左边（包括mid）
                right = mid
        
        return left


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    nums1 = [1, 2, 3, 1]
    result1 = solution.findPeakElement(nums1)
    print(f"输入: nums = {nums1}")
    print(f"输出: {result1}")
    print()
    
    # 测试用例2
    nums2 = [1, 2, 1, 3, 5, 6, 4]
    result2 = solution.findPeakElement(nums2)
    print(f"输入: nums = {nums2}")
    print(f"输出: {result2}")
