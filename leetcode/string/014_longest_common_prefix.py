"""
14. 最长公共前缀 (Longest Common Prefix)

难度：简单

题目描述：
编写一个函数来查找字符串数组中的最长公共前缀。
如果不存在公共前缀，返回空字符串 ""。

示例 1：
输入：strs = ["flower","flow","flight"]
输出："fl"

示例 2：
输入：strs = ["dog","racecar","car"]
输出：""
解释：输入不存在公共前缀。

提示：
1 <= strs.length <= 200
0 <= strs[i].length <= 200
strs[i] 仅由小写英文字母组成
"""

from typing import List


class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        """
        纵向扫描：逐列比较所有字符串的字符
        时间复杂度：O(S)，S是所有字符串的字符总数
        空间复杂度：O(1)
        """
        if not strs:
            return ""
        
        # 以第一个字符串为基准
        for i in range(len(strs[0])):
            char = strs[0][i]
            
            # 比较其他字符串的对应位置
            for j in range(1, len(strs)):
                # 如果到达某个字符串的末尾或字符不匹配
                if i >= len(strs[j]) or strs[j][i] != char:
                    return strs[0][:i]
        
        return strs[0]


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    strs1 = ["flower", "flow", "flight"]
    result1 = solution.longestCommonPrefix(strs1)
    print(f"输入: strs = {strs1}")
    print(f"输出: \"{result1}\"")
    print()
    
    # 测试用例2
    strs2 = ["dog", "racecar", "car"]
    result2 = solution.longestCommonPrefix(strs2)
    print(f"输入: strs = {strs2}")
    print(f"输出: \"{result2}\"")
