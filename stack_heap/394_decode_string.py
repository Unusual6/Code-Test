"""
394. 字符串解码 (Decode String)
难度：中等
给定一个经过编码的字符串，返回它解码后的字符串。
编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。
示例：输入：s = "3[a]2[bc]"  输出："aaabcbc"
"""

class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        current_num = 0
        current_str = ""
        for char in s:
            if char == '[':
                stack.append((current_str, current_num))
                current_str = ""
                current_num = 0
            elif char == ']':
                prev_str, num = stack.pop()
                current_str = prev_str + num * current_str
            elif char.isdigit():
                current_num = current_num * 10 + int(char)
            else:
                current_str += char
        return current_str
