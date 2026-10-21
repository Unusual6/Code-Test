"""
49. 字母异位词分组 (Group Anagrams)

难度：中等

题目描述：
给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
字母异位词 是由重新排列源单词的所有字母得到的一个新单词。

示例 1:
输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
输出: [["bat"],["nat","tan"],["ate","eat","tea"]]

示例 2:
输入: strs = [""]
输出: [[""]]

示例 3:
输入: strs = ["a"]
输出: [["a"]]

提示：
1 <= strs.length <= 10^4
0 <= strs[i].length <= 100
strs[i] 仅包含小写字母
"""

from typing import List
from collections import defaultdict


class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """
        使用排序后的字符串作为键
        时间复杂度：O(n * k * log(k))，其中 n 是字符串数量，k 是字符串最大长度
        空间复杂度：O(n * k)
        """
        anagram_map = defaultdict(list)
        
        for s in strs:
            # 将字符串排序后作为键
            key = tuple(sorted(s))
            anagram_map[key].append(s)
        
        return list(anagram_map.values())


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    strs1 = ["eat", "tea", "tan", "ate", "nat", "bat"]
    result1 = solution.groupAnagrams(strs1)
    print(f"输入: strs = {strs1}")
    print(f"输出: {result1}")
    print()
    
    # 测试用例2
    strs2 = [""]
    result2 = solution.groupAnagrams(strs2)
    print(f"输入: strs = {strs2}")
    print(f"输出: {result2}")
    print()
    
    # 测试用例3
    strs3 = ["a"]
    result3 = solution.groupAnagrams(strs3)
    print(f"输入: strs = {strs3}")
    print(f"输出: {result3}")
