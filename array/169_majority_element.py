"""
169. 多数元素 (Majority Element)

难度：简单

题目描述：
给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。
你可以假设数组是非空的，并且给定的数组总是存在多数元素。

示例 1：
输入：nums = [3,2,3]
输出：3

示例 2：
输入：nums = [2,2,1,1,1,2,2]
输出：2

提示：
n == nums.length
1 <= n <= 5 * 10^4
-10^9 <= nums[i] <= 10^9

进阶：尝试设计时间复杂度为 O(n)、空间复杂度为 O(1) 的算法解决此问题。
"""

from typing import List


class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        """
        Boyer-Moore 投票算法
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        candidate = None
        count = 0
        
        for num in nums:
            if count == 0:
                candidate = num
            
            if num == candidate:
                count += 1
            else:
                count -= 1
        
        return candidate


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    nums1 = [3, 2, 3]
    result1 = solution.majorityElement(nums1)
    print(f"输入: nums = {nums1}")
    print(f"输出: {result1}")
    print()
    
    # 测试用例2
    nums2 = [2, 2, 1, 1, 1, 2, 2]
    result2 = solution.majorityElement(nums2)
    print(f"输入: nums = {nums2}")
    print(f"输出: {result2}")
