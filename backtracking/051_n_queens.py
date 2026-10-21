"""
51. N 皇后
难度：困难
按照国际象棋的规则，皇后可以攻击与之处在同一行或同一列或同一斜线上的棋子。
n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
示例：输入：n = 4  输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
"""
from typing import List

class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        result = []
        board = [['.'] * n for _ in range(n)]
        self._backtrack(board, 0, n, result)
        return result
    
    def _backtrack(self, board, row, n, result):
        if row == n:
            result.append([''.join(r) for r in board])
            return
        for col in range(n):
            if self._isValid(board, row, col, n):
                board[row][col] = 'Q'
                self._backtrack(board, row + 1, n, result)
                board[row][col] = '.'
    
    def _isValid(self, board, row, col, n):
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1
        return True
