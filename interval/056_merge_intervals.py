"""
56. 合并区间 (Merge Intervals)

难度：中等

题目描述：
以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。
请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。

示例 1：
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].

示例 2：
输入：intervals = [[1,4],[4,5]]
输出：[[1,5]]
解释：区间 [1,4] 和 [4,5] 可被视为重叠区间。

提示：
1 <= intervals.length <= 10^4
intervals[i].length == 2
0 <= starti <= endi <= 10^4
"""

from typing import List


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        先按起始位置排序，然后合并重叠区间
        时间复杂度：O(n log n)
        空间复杂度：O(n)
        """
        if not intervals:
            return []
        
        # 按起始位置排序
        intervals.sort(key=lambda x: x[0])
        
        merged = [intervals[0]]
        
        for i in range(1, len(intervals)):
            current = intervals[i]
            last = merged[-1]
            
            # 如果当前区间的起始位置小于等于上一个合并区间的结束位置，说明有重叠
            if current[0] <= last[1]:
                # 合并区间，取较大的结束位置
                merged[-1][1] = max(last[1], current[1])
            else:
                merged.append(current)
        
        return merged


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    intervals1 = [[1, 3], [2, 6], [8, 10], [15, 18]]
    result1 = solution.merge(intervals1)
    print(f"输入: intervals = {intervals1}")
    print(f"输出: {result1}")
    print()
    
    # 测试用例2
    intervals2 = [[1, 4], [4, 5]]
    result2 = solution.merge(intervals2)
    print(f"输入: intervals = {intervals2}")
    print(f"输出: {result2}")
