"""
31. 下一个排列 (Next Permutation)

难度：中等

题目描述：
整数数组的一个 排列  就是将其所有成员以序列/线性顺序重新排列。
例如，arr = [1,2,3] 的排列有：[1,2,3]、[1,3,2]、[2,1,3]、[2,3,1]、[3,1,2]、[3,2,1]。
整数数组的 下一个排列 是指其整数的下一个字典序更大的排列。
如果不存在下一个更大的排列，则将数组重排为字典序最小的排列（升序排列）。
必须 原地 修改，只允许使用额外常数空间。

示例 1：
输入：nums = [1,2,3]
输出：[1,3,2]

示例 2：
输入：nums = [3,2,1]
输出：[1,2,3]

示例 3：
输入：nums = [1,1,5]
输出：[1,5,1]

提示：
1 <= nums.length <= 100
0 <= nums[i] <= 100
"""

from typing import List


class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        三步法：
        1. 从右向左找到第一个递增的位置 i（nums[i] < nums[i+1]）
        2. 从右向左找到第一个大于 nums[i] 的数 nums[j]，交换
        3. 反转 i+1 之后的部分
        时间复杂度：O(n)
        空间复杂度：O(1)
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        
        # 步骤1：从右向左找到第一个递减的位置
        i = n - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        
        if i >= 0:
            # 步骤2：从右向左找到第一个大于 nums[i] 的数
            j = n - 1
            while j >= 0 and nums[j] <= nums[i]:
                j -= 1
            # 交换 nums[i] 和 nums[j]
            nums[i], nums[j] = nums[j], nums[i]
        
        # 步骤3：反转 i+1 之后的部分
        left, right = i + 1, n - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    nums1 = [1, 2, 3]
    print(f"输入: nums = {nums1}")
    solution.nextPermutation(nums1)
    print(f"输出: {nums1}")
    print()
    
    # 测试用例2
    nums2 = [3, 2, 1]
    print(f"输入: nums = {nums2}")
    solution.nextPermutation(nums2)
    print(f"输出: {nums2}")
    print()
    
    # 测试用例3
    nums3 = [1, 1, 5]
    print(f"输入: nums = {nums3}")
    solution.nextPermutation(nums3)
    print(f"输出: {nums3}")
