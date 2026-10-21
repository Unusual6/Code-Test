"""
11. 盛最多水的容器 (Container With Most Water)

难度：中等

题目描述：
给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。
找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
返回容器可以储存的最大水量。
说明：你不能倾斜容器。

示例 1：
输入：[1,8,6,2,5,4,8,3,7]
输出：49 
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够储存水量的最大值是 49。

示例 2：
输入：height = [1,1]
输出：1

提示：
n == height.length
2 <= n <= 10^5
0 <= height[i] <= 10^4
"""

from typing import List


class Solution:
    def maxArea(self, height: List[int]) -> int:
        """
        双指针法：左右指针向中间移动，每次移动较短的那条线
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        left = 0
        right = len(height) - 1
        max_water = 0
        
        while left < right:
            # 计算当前容器的水量
            width = right - left
            h = min(height[left], height[right])
            water = width * h
            max_water = max(max_water, water)
            
            # 移动较短的那条线
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        
        return max_water


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    height1 = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    result1 = solution.maxArea(height1)
    print(f"输入: height = {height1}")
    print(f"输出: {result1}")
    print()
    
    # 测试用例2
    height2 = [1, 1]
    result2 = solution.maxArea(height2)
    print(f"输入: height = {height2}")
    print(f"输出: {result2}")
