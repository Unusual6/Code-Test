"""
22. 括号生成
难度：中等
数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且有效的括号组合。
示例：输入：n = 3  输出：["((()))","(()())","(())()","()(())","()()()"]
"""
from typing import List

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        result = []
        self._backtrack(n, 0, 0, "", result)
        return result
    
    def _backtrack(self, n, open_count, close_count, path, result):
        if len(path) == 2 * n:
            result.append(path)
            return
        if open_count < n:
            self._backtrack(n, open_count + 1, close_count, path + "(", result)
        if close_count < open_count:
            self._backtrack(n, open_count, close_count + 1, path + ")", result)
