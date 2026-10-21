"""
347. 前 K 个高频元素 (Top K Frequent Elements)
难度：中等
给你一个整数数组 nums 和一个整数 k，请你返回其中出现频率前 k 高的元素。
示例：输入：nums = [1,1,1,2,2,3], k = 2  输出：[1,2]
"""
from typing import List
from collections import Counter
import heapq

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = Counter(nums)
        return heapq.nlargest(k, count.keys(), key=count.get)
