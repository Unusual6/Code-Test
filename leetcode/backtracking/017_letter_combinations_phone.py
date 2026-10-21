"""
17. 电话号码的字母组合
难度：中等
给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。
示例：输入：digits = "23"  输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
"""
from typing import List

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        phone = {'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
        result = []
        self._backtrack(digits, 0, "", phone, result)
        return result
    
    def _backtrack(self, digits, index, path, phone, result):
        if index == len(digits):
            result.append(path)
            return
        for char in phone[digits[index]]:
            self._backtrack(digits, index + 1, path + char, phone, result)
