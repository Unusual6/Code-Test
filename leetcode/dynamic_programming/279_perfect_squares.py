"""
279. 完全平方数
难度：中等
给你一个整数 n，返回和为 n 的完全平方数的最少数量。
示例：输入：n = 12  输出：3 (12 = 4 + 4 + 4)
"""

class Solution:
    def numSquares(self, n: int) -> int:
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            j = 1
            while j * j <= i:
                dp[i] = min(dp[i], dp[i - j * j] + 1)
                j += 1
        return dp[n]
