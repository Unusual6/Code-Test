"""
15. 三数之和 (3Sum)

难度：中等

题目描述：
给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，
同时还满足 nums[i] + nums[j] + nums[k] == 0 。请你返回所有和为 0 且不重复的三元组。
注意：答案中不可以包含重复的三元组。

示例 1：
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]

示例 2：
输入：nums = [0,1,1]
输出：[]

示例 3：
输入：nums = [0,0,0]
输出：[[0,0,0]]

提示：
3 <= nums.length <= 3000
-10^5 <= nums[i] <= 10^5
"""

from typing import List


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
        排序 + 双指针
        时间复杂度：O(n^2)
        空间复杂度：O(1) 或 O(n) 取决于排序算法
        """
        nums.sort()
        result = []
        n = len(nums)
        
        for i in range(n - 2):
            # 如果当前数字大于0，则三数之和一定大于0
            if nums[i] > 0:
                break
            
            # 跳过重复的第一个数
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            
            left = i + 1
            right = n - 1
            
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                
                if total < 0:
                    left += 1
                elif total > 0:
                    right -= 1
                else:
                    result.append([nums[i], nums[left], nums[right]])
                    
                    # 跳过重复的第二个数
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    # 跳过重复的第三个数
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
        
        return result


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    nums1 = [-1, 0, 1, 2, -1, -4]
    result1 = solution.threeSum(nums1)
    print(f"输入: nums = {nums1}")
    print(f"输出: {result1}")
    print()
    
    # 测试用例2
    nums2 = [0, 1, 1]
    result2 = solution.threeSum(nums2)
    print(f"输入: nums = {nums2}")
    print(f"输出: {result2}")
    print()
    
    # 测试用例3
    nums3 = [0, 0, 0]
    result3 = solution.threeSum(nums3)
    print(f"输入: nums = {nums3}")
    print(f"输出: {result3}")
