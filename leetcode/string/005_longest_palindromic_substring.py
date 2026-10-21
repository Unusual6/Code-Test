"""
5. 最长回文子串 (Longest Palindromic Substring)

难度：中等

题目描述：
给你一个字符串 s，找到 s 中最长的回文子串。
如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。

示例 1：
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。

示例 2：
输入：s = "cbbd"
输出："bb"

提示：
1 <= s.length <= 1000
s 仅由数字和英文字母组成
"""


class Solution:
    def longestPalindrome(self, s: str) -> str:
        """
        中心扩展法：遍历每个字符作为中心，向两边扩展
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        if not s:
            return ""
        
        start = 0
        max_len = 1
        
        def expandAroundCenter(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return left + 1, right - 1
        
        for i in range(len(s)):
            # 奇数长度的回文
            left1, right1 = expandAroundCenter(i, i)
            # 偶数长度的回文
            left2, right2 = expandAroundCenter(i, i + 1)
            
            if right1 - left1 + 1 > max_len:
                start = left1
                max_len = right1 - left1 + 1
            
            if right2 - left2 + 1 > max_len:
                start = left2
                max_len = right2 - left2 + 1
        
        return s[start:start + max_len]


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    s1 = "babad"
    result1 = solution.longestPalindrome(s1)
    print(f"输入: s = \"{s1}\"")
    print(f"输出: \"{result1}\"")
    print()
    
    # 测试用例2
    s2 = "cbbd"
    result2 = solution.longestPalindrome(s2)
    print(f"输入: s = \"{s2}\"")
    print(f"输出: \"{result2}\"")
