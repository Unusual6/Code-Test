"""
54. 螺旋矩阵 (Spiral Matrix)

难度：中等

题目描述：
给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。

示例 1：
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]

示例 2：
输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]

提示：
m == matrix.length
n == matrix[i].length
1 <= m, n <= 10
-100 <= matrix[i][j] <= 100
"""

from typing import List


class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        """
        模拟螺旋遍历：设定上下左右边界，依次遍历
        时间复杂度：O(mn)
        空间复杂度：O(1)
        """
        if not matrix:
            return []
        
        m, n = len(matrix), len(matrix[0])
        result = []
        
        top, bottom = 0, m - 1
        left, right = 0, n - 1
        
        while top <= bottom and left <= right:
            # 从左到右遍历上边界
            for j in range(left, right + 1):
                result.append(matrix[top][j])
            top += 1
            
            # 从上到下遍历右边界
            for i in range(top, bottom + 1):
                result.append(matrix[i][right])
            right -= 1
            
            if top <= bottom:
                # 从右到左遍历下边界
                for j in range(right, left - 1, -1):
                    result.append(matrix[bottom][j])
                bottom -= 1
            
            if left <= right:
                # 从下到上遍历左边界
                for i in range(bottom, top - 1, -1):
                    result.append(matrix[i][left])
                left += 1
        
        return result


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    matrix1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result1 = solution.spiralOrder(matrix1)
    print(f"输入: matrix = {matrix1}")
    print(f"输出: {result1}")
    print()
    
    # 测试用例2
    matrix2 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    result2 = solution.spiralOrder(matrix2)
    print(f"输入: matrix = {matrix2}")
    print(f"输出: {result2}")
