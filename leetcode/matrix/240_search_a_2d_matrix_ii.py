"""
240. 搜索二维矩阵 II (Search a 2D Matrix II)

难度：中等

题目描述：
编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
每行的元素从左到右升序排列。
每列的元素从上到下升序排列。

示例 1：
输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
输出：true

示例 2：
输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
输出：false

提示：
m == matrix.length
n == matrix[i].length
1 <= n, m <= 300
-10^9 <= matrix[i][j] <= 10^9
每行的所有元素从左到右按 非递减顺序 排列
每列的所有元素从上到下按 非递减顺序 排列
-10^9 <= target <= 10^9
"""

from typing import List


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        从右上角开始搜索：如果当前值大于目标，向左移动；如果小于目标，向下移动
        时间复杂度：O(m + n)
        空间复杂度：O(1)
        """
        if not matrix or not matrix[0]:
            return False
        
        m, n = len(matrix), len(matrix[0])
        row, col = 0, n - 1  # 从右上角开始
        
        while row < m and col >= 0:
            current = matrix[row][col]
            
            if current == target:
                return True
            elif current > target:
                col -= 1  # 当前值太大，向左移动
            else:
                row += 1  # 当前值太小，向下移动
        
        return False


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    matrix1 = [
        [1, 4, 7, 11, 15],
        [2, 5, 8, 12, 19],
        [3, 6, 9, 16, 22],
        [10, 13, 14, 17, 24],
        [18, 21, 23, 26, 30]
    ]
    target1 = 5
    result1 = solution.searchMatrix(matrix1, target1)
    print(f"输入: matrix = {matrix1}, target = {target1}")
    print(f"输出: {result1}")
    print()
    
    # 测试用例2
    target2 = 20
    result2 = solution.searchMatrix(matrix1, target2)
    print(f"输入: matrix = {matrix1}, target = {target2}")
    print(f"输出: {result2}")
