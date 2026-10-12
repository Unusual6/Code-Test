"""
32. 最长有效括号
难度：困难
给你一个只包含 '(' 和 ')' 的字符串，找出并返回最长的有效括号子串的长度。
示例：输入：s = "(()"  输出：2
"""

class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack = [-1]
        max_len = 0
        for i, char in enumerate(s):
            if char == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    max_len = max(max_len, i - stack[-1])
        return max_len
