"""
118. 杨辉三角
难度：简单
给定一个非负整数 numRows，生成「杨辉三角」的前 numRows 行。
示例：输入: numRows = 5  输出: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
"""
from typing import List

class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        result = [[1]]
        for i in range(1, numRows):
            row = [1]
            for j in range(1, i):
                row.append(result[i-1][j-1] + result[i-1][j])
            row.append(1)
            result.append(row)
        return result
