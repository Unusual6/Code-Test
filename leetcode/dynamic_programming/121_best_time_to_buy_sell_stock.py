"""
121. 买卖股票的最佳时机 (Best Time to Buy and Sell Stock)
难度：简单
给定一个数组 prices，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
你只能选择某一天买入这只股票，并选择在未来的某一个不同的日子卖出。设计一个算法来计算你所能获取的最大利润。
示例：输入：[7,1,5,3,6,4]  输出：5
解释：在第 2 天买入(价格=1)，在第 5 天卖出(价格=6)，利润=6-1=5。
"""
from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_price = float('inf')
        max_profit = 0
        for price in prices:
            min_price = min(min_price, price)
            max_profit = max(max_profit, price - min_price)
        return max_profit
