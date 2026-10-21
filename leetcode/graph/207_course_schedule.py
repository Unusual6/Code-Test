"""
207. 课程表
难度：中等
判断是否可能完成所有课程的学习。
示例：输入：numCourses = 2, prerequisites = [[1,0]]  输出：true
"""
from typing import List
from collections import deque

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        in_degree = [0] * numCourses
        graph = [[] for _ in range(numCourses)]
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1
        queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
        count = 0
        while queue:
            node = queue.popleft()
            count += 1
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return count == numCourses
