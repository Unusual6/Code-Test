"""
62. 不同路径 (Unique Paths)
难度：中等
一个机器人位于一个 m x n 网格的左上角。机器人每次只能向下或者向右移动一步。
机器人试图达到网格的右下角。问总共有多少条不同的路径？
示例：输入：m = 3, n = 7  输出：28
"""

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1] * n for _ in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]
