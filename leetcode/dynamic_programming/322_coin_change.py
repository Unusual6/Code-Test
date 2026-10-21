"""
322. 零钱兑换
难度：中等
给你一个整数数组 coins，表示不同面额的硬币；以及一个整数 amount，表示总金额。
计算并返回可以凑成总金额所需的最少的硬币个数。
示例：输入：coins = [1, 2, 5], amount = 11  输出：3
"""
from typing import List

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[amount] if dp[amount] != float('inf') else -1
