"""
74. 搜索二维矩阵
难度：中等
给你一个满足下述两个属性的 m x n 整数矩阵：每行整数按升序排列，每行的第一个整数大于前一行的最后一个整数。
示例：输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3  输出：true
"""
from typing import List

class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        left, right = 0, m * n - 1
        while left <= right:
            mid = (left + right) // 2
            val = matrix[mid // n][mid % n]
            if val == target:
                return True
            elif val < target:
                left = mid + 1
            else:
                right = mid - 1
        return False
