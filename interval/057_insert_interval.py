"""
57. 插入区间 (Insert Interval)

难度：中等

题目描述：
给你一个 无重叠的 ，按照区间起始端点排序的区间列表 intervals，其中 intervals[i] = [starti, endi] 
表示一个起始于 starti 且结束于 endi 的区间，这些区间按照起始端点升序排列。
你还需要再插入一个区间 newInterval = [start, end] 。
请将区间列表中的 newInterval 插入，使得 intervals 依然按照起始端点升序排列，
且区间之间依然没有重叠（如果有必要的话，可以进行合并）。
返回插入后 intervals 。

示例 1：
输入：intervals = [[1,3],[6,9]], newInterval = [2,5]
输出：[[1,5],[6,9]]

示例 2：
输入：intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
输出：[[1,2],[3,10],[12,16]]
解释：这是因为新的区间 [4, 8] 与 [3,5],[6,7],[8,10] 重叠。

提示：
0 <= intervals.length <= 10^4
intervals[i].length == 2
0 <= starti <= endi <= 10^5
intervals 根据 starti 按 升序 排列
newInterval.length == 2
0 <= start <= end <= 10^5
"""

from typing import List


class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        """
        一次遍历，分三部分处理：不重叠的左边、需要合并的部分、不重叠的右边
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        result = []
        i = 0
        n = len(intervals)
        
        # 添加所有在 newInterval 之前且不重叠的区间
        while i < n and intervals[i][1] < newInterval[0]:
            result.append(intervals[i])
            i += 1
        
        # 合并所有重叠的区间
        while i < n and intervals[i][0] <= newInterval[1]:
            newInterval[0] = min(newInterval[0], intervals[i][0])
            newInterval[1] = max(newInterval[1], intervals[i][1])
            i += 1
        
        result.append(newInterval)
        
        # 添加所有在 newInterval 之后且不重叠的区间
        while i < n:
            result.append(intervals[i])
            i += 1
        
        return result


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    intervals1 = [[1, 3], [6, 9]]
    newInterval1 = [2, 5]
    result1 = solution.insert(intervals1, newInterval1)
    print(f"输入: intervals = {intervals1}, newInterval = {newInterval1}")
    print(f"输出: {result1}")
    print()
    
    # 测试用例2
    intervals2 = [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]]
    newInterval2 = [4, 8]
    result2 = solution.insert(intervals2, newInterval2)
    print(f"输入: intervals = {intervals2}, newInterval = {newInterval2}")
    print(f"输出: {result2}")
