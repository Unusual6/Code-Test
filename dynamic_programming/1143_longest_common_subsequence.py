"""
1143. 最长公共子序列
难度：中等
给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。
示例：输入：text1 = "abcde", text2 = "ace"  输出：3
"""

class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
