"""
3. 无重复字符的最长子串 (Longest Substring Without Repeating Characters)

难度：中等

题目描述：
给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。

示例 1:
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

示例 2:
输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。

示例 3:
输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。

提示：
0 <= s.length <= 5 * 10^4
s 由英文字母、数字、符号和空格组成
"""


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        """
        滑动窗口：使用哈希集合记录窗口中的字符
        时间复杂度：O(n)
        空间复杂度：O(min(n, m))，m是字符集大小
        """
        char_set = set()
        left = 0
        max_length = 0
        
        for right in range(len(s)):
            # 如果字符已在窗口中，移动左指针
            while s[right] in char_set:
                char_set.remove(s[left])
                left += 1
            
            char_set.add(s[right])
            max_length = max(max_length, right - left + 1)
        
        return max_length


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    s1 = "abcabcbb"
    result1 = solution.lengthOfLongestSubstring(s1)
    print(f"输入: s = \"{s1}\"")
    print(f"输出: {result1}")
    print()
    
    # 测试用例2
    s2 = "bbbbb"
    result2 = solution.lengthOfLongestSubstring(s2)
    print(f"输入: s = \"{s2}\"")
    print(f"输出: {result2}")
    print()
    
    # 测试用例3
    s3 = "pwwkew"
    result3 = solution.lengthOfLongestSubstring(s3)
    print(f"输入: s = \"{s3}\"")
    print(f"输出: {result3}")
