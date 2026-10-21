"""
73. 矩阵置零 (Set Matrix Zeroes)

难度：中等

题目描述：
给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。

示例 1：
输入：matrix = [[1,1,1],[1,0,1],[1,1,1]]
输出：[[1,0,1],[0,0,0],[1,0,1]]

示例 2：
输入：matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
输出：[[0,0,0,0],[0,4,5,0],[0,3,1,0]]

提示：
m == matrix.length
n == matrix[0].length
1 <= m, n <= 200
-2^31 <= matrix[i][j] <= 2^31 - 1

进阶：
一个直观的解决方案是使用  O(mn) 的额外空间，但这并不是一个好的解决方案。
一个简单的改进方案是使用 O(m + n) 的额外空间，但这仍然不是最好的解决方案。
你能想出一个仅使用常量空间的解决方案吗？
"""

from typing import List


class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        使用第一行和第一列作为标记，同时用两个变量记录第一行和第一列是否需要置零
        时间复杂度：O(mn)
        空间复杂度：O(1)
        Do not return anything, modify matrix in-place instead.
        """
        m, n = len(matrix), len(matrix[0])
        
        # 检查第一行和第一列是否需要置零
        first_row_zero = any(matrix[0][j] == 0 for j in range(n))
        first_col_zero = any(matrix[i][0] == 0 for i in range(m))
        
        # 使用第一行和第一列作为标记
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
        
        # 根据标记置零
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        
        # 处理第一行
        if first_row_zero:
            for j in range(n):
                matrix[0][j] = 0
        
        # 处理第一列
        if first_col_zero:
            for i in range(m):
                matrix[i][0] = 0


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    matrix1 = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    print(f"输入: matrix = {matrix1}")
    solution.setZeroes(matrix1)
    print(f"输出: {matrix1}")
    print()
    
    # 测试用例2
    matrix2 = [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]
    print(f"输入: matrix = {matrix2}")
    solution.setZeroes(matrix2)
    print(f"输出: {matrix2}")
