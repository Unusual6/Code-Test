"""
1. 两数之和 (Two Sum)

难度：简单

题目描述：
给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target 的那 两个 整数，并返回它们的数组下标。
你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。
你可以按任意顺序返回答案。

示例 1：
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

示例 2：
输入：nums = [3,2,4], target = 6
输出：[1,2]

示例 3：
输入：nums = [3,3], target = 6
输出：[0,1]

提示：
2 <= nums.length <= 10^4
-10^9 <= nums[i] <= 10^9
-10^9 <= target <= 10^9
只会存在一个有效答案

进阶：你可以想出一个时间复杂度小于 O(n^2) 的算法吗？
"""

from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        使用哈希表存储已遍历的数字及其索引
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        num_map = {}  # 存储数字到索引的映射
        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_map:
                return [num_map[complement], i]
            num_map[num] = i
        return []


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    nums1 = [2, 7, 11, 15]
    target1 = 9
    result1 = solution.twoSum(nums1, target1)
    print(f"输入: nums = {nums1}, target = {target1}")
    print(f"输出: {result1}")
    print()
    
    # 测试用例2
    nums2 = [3, 2, 4]
    target2 = 6
    result2 = solution.twoSum(nums2, target2)
    print(f"输入: nums = {nums2}, target = {target2}")
    print(f"输出: {result2}")
    print()
    
    # 测试用例3
    nums3 = [3, 3]
    target3 = 6
    result3 = solution.twoSum(nums3, target3)
    print(f"输入: nums = {nums3}, target = {target3}")
    print(f"输出: {result3}")
