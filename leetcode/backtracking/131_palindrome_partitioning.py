"""
131. 分割回文串
难度：中等
给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是回文串。返回 s 所有可能的分割方案。
示例：输入：s = "aab"  输出：[["a","a","b"],["aa","b"]]
"""
from typing import List

class Solution:
    def partition(self, s: str) -> List[List[str]]:
        result = []
        self._backtrack(s, 0, [], result)
        return result
    
    def _backtrack(self, s, start, path, result):
        if start == len(s):
            result.append(path[:])
            return
        for end in range(start + 1, len(s) + 1):
            if self._isPalindrome(s, start, end - 1):
                path.append(s[start:end])
                self._backtrack(s, end, path, result)
                path.pop()
    
    def _isPalindrome(self, s, left, right):
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True
