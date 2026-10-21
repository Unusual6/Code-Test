"""
27. 移除元素 (Remove Element)

难度：简单

题目描述：
给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

示例 1：
输入：nums = [3,2,2,3], val = 3
输出：2, nums = [2,2]
解释：函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。

示例 2：
输入：nums = [0,1,2,2,3,0,4,2], val = 2
输出：5, nums = [0,1,4,0,3]
解释：函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。

提示：
0 <= nums.length <= 100
0 <= nums[i] <= 50
0 <= val <= 100
"""

from typing import List


class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        """
        双指针法：快指针遍历数组，慢指针记录不等于 val 的元素
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        slow = 0
        
        for fast in range(len(nums)):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
        
        return slow


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    nums1 = [3, 2, 2, 3]
    val1 = 3
    print(f"输入: nums = {nums1}, val = {val1}")
    length1 = solution.removeElement(nums1, val1)
    print(f"输出: length = {length1}, nums = {nums1[:length1]}")
    print()
    
    # 测试用例2
    nums2 = [0, 1, 2, 2, 3, 0, 4, 2]
    val2 = 2
    print(f"输入: nums = {nums2}, val = {val2}")
    length2 = solution.removeElement(nums2, val2)
    print(f"输出: length = {length2}, nums = {nums2[:length2]}")
