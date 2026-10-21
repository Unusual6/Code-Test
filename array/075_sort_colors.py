"""
75. 颜色分类 (Sort Colors)

难度：中等

题目描述：
给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地 对它们进行排序，使得红色、白色和蓝色相邻，
并且按照红色、白色、蓝色顺序排列。
我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
必须在不使用库内置的 sort 函数的情况下解决此问题。
你能想出一个仅使用常数空间的一趟扫描算法吗？

示例 1：
输入：nums = [2,0,2,1,1,0]
输出：[0,0,1,1,2,2]

示例 2：
输入：nums = [2,0,1]
输出：[0,1,2]

提示：
n == nums.length
1 <= n <= 300
nums[i] 为 0、1 或 2

进阶：
你可以使用仅使用常数空间的一趟扫描算法来解决这个问题吗？
"""

from typing import List


class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        三指针法（荷兰国旗问题）：
        - left 指向下一个 0 应该放置的位置
        - right 指向下一个 2 应该放置的位置
        - current 遍历数组
        时间复杂度：O(n)
        空间复杂度：O(1)
        Do not return anything, modify nums in-place instead.
        """
        left = 0
        right = len(nums) - 1
        current = 0
        
        while current <= right:
            if nums[current] == 0:
                nums[left], nums[current] = nums[current], nums[left]
                left += 1
                current += 1
            elif nums[current] == 2:
                nums[right], nums[current] = nums[current], nums[right]
                right -= 1
                # 注意：这里 current 不增加，因为交换过来的元素还需要检查
            else:
                current += 1


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    nums1 = [2, 0, 2, 1, 1, 0]
    print(f"输入: nums = {nums1}")
    solution.sortColors(nums1)
    print(f"输出: {nums1}")
    print()
    
    # 测试用例2
    nums2 = [2, 0, 1]
    print(f"输入: nums = {nums2}")
    solution.sortColors(nums2)
    print(f"输出: {nums2}")
