"""
42. 接雨水 (Trapping Rain Water)

难度：困难

题目描述：
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

示例 1：
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。

示例 2：
输入：height = [4,2,0,3,2,5]
输出：9

提示：
n == height.length
1 <= n <= 2 * 10^4
0 <= height[i] <= 10^5
"""

from typing import List


class Solution:
    def trap(self, height: List[int]) -> int:
        """
        双指针法：左右指针向中间移动，维护左右最大值
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not height:
            return 0
        
        left = 0
        right = len(height) - 1
        left_max = height[left]
        right_max = height[right]
        water = 0
        
        while left < right:
            if left_max < right_max:
                left += 1
                left_max = max(left_max, height[left])
                water += left_max - height[left]
            else:
                right -= 1
                right_max = max(right_max, height[right])
                water += right_max - height[right]
        
        return water


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    height1 = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    result1 = solution.trap(height1)
    print(f"输入: height = {height1}")
    print(f"输出: {result1}")
    print()
    
    # 测试用例2
    height2 = [4, 2, 0, 3, 2, 5]
    result2 = solution.trap(height2)
    print(f"输入: height = {height2}")
    print(f"输出: {result2}")
