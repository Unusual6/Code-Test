"""
79. 单词搜索
难度：中等
给定一个 m x n 二维字符网格 board 和一个字符串单词 word。如果 word 存在于网格中，返回 true。
示例：输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"  输出：true
"""
from typing import List

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m, n = len(board), len(board[0])
        def dfs(i, j, k):
            if k == len(word):
                return True
            if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[k]:
                return False
            temp = board[i][j]
            board[i][j] = '#'
            found = (dfs(i+1, j, k+1) or dfs(i-1, j, k+1) or 
                     dfs(i, j+1, k+1) or dfs(i, j-1, k+1))
            board[i][j] = temp
            return found
        
        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0):
                    return True
        return False
