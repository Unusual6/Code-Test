"""
128. 最长连续序列 (Longest Consecutive Sequence)

难度：中等

题目描述：
给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
请你设计并实现时间复杂度为 O(n) 的算法解决此问题。

示例 1：
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。

示例 2：
输入：nums = [0,3,7,2,5,8,4,6,0,1]
输出：9

提示：
0 <= nums.length <= 10^5
-10^9 <= nums[i] <= 10^9
"""

from typing import List


class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        """
        使用哈希集合，只从序列起始位置开始计数
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        num_set = set(nums)
        longest_streak = 0
        
        for num in num_set:
            # 只有当 num-1 不在集合中时，才开始一个新的序列
            if num - 1 not in num_set:
                current_num = num
                current_streak = 1
                
                # 向后查找连续的数
                while current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1
                
                longest_streak = max(longest_streak, current_streak)
        
        return longest_streak


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    nums1 = [100, 4, 200, 1, 3, 2]
    result1 = solution.longestConsecutive(nums1)
    print(f"输入: nums = {nums1}")
    print(f"输出: {result1}")
    print()
    
    # 测试用例2
    nums2 = [0, 3, 7, 2, 5, 8, 4, 6, 0, 1]
    result2 = solution.longestConsecutive(nums2)
    print(f"输入: nums = {nums2}")
    print(f"输出: {result2}")
