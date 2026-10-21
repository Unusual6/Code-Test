"""
215. 数组中的第K个最大元素 (Kth Largest Element in an Array)

难度：中等

题目描述：
给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。

示例 1:
输入: [3,2,1,5,6,4], k = 2
输出: 5

示例 2:
输入: [3,2,3,1,2,4,5,5,6], k = 4
输出: 4

提示：
1 <= k <= nums.length <= 10^5
-10^4 <= nums[i] <= 10^4
"""

from typing import List
import random


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        """
        快速选择算法（基于快排思想）
        时间复杂度：O(n) 平均
        空间复杂度：O(1)
        """
        def quickSelect(left, right, k_smallest):
            if left == right:
                return nums[left]
            
            # 随机选择pivot
            pivot_index = random.randint(left, right)
            nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
            
            # 分区
            pivot = nums[right]
            store_index = left
            
            for i in range(left, right):
                if nums[i] < pivot:
                    nums[store_index], nums[i] = nums[i], nums[store_index]
                    store_index += 1
            
            nums[store_index], nums[right] = nums[right], nums[store_index]
            
            if k_smallest == store_index:
                return nums[store_index]
            elif k_smallest < store_index:
                return quickSelect(left, store_index - 1, k_smallest)
            else:
                return quickSelect(store_index + 1, right, k_smallest)
        
        # 第k大等于第(n-k)小
        return quickSelect(0, len(nums) - 1, len(nums) - k)


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    nums1 = [3, 2, 1, 5, 6, 4]
    k1 = 2
    result1 = solution.findKthLargest(nums1, k1)
    print(f"输入: nums = {nums1}, k = {k1}")
    print(f"输出: {result1}")
    print()
    
    # 测试用例2
    nums2 = [3, 2, 3, 1, 2, 4, 5, 5, 6]
    k2 = 4
    result2 = solution.findKthLargest(nums2, k2)
    print(f"输入: nums = {nums2}, k = {k2}")
    print(f"输出: {result2}")
