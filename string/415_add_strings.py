"""
415. 字符串相加 (Add Strings)

难度：简单

题目描述：
给定两个字符串形式的非负整数 num1 和 num2 ，计算它们的和并同样以字符串形式返回。
你必须在不使用任何内置库用于处理大整数（例如 BigInteger）的情况下解决此问题。
也不能直接将输入的字符串转换为整数形式。

示例 1：
输入：num1 = "11", num2 = "123"
输出："134"

示例 2：
输入：num1 = "456", num2 = "77"
输出："533"

示例 3：
输入：num1 = "0", num2 = "0"
输出："0"

提示：
1 <= num1.length, num2.length <= 10^4
num1 和 num2 仅由数字组成
num1 和 num2 都不含任何前导零
"""


class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        """
        模拟竖式加法：从后往前逐位相加
        时间复杂度：O(max(n1, n2))
        空间复杂度：O(1)
        """
        result = []
        carry = 0
        i, j = len(num1) - 1, len(num2) - 1
        
        while i >= 0 or j >= 0 or carry:
            digit1 = int(num1[i]) if i >= 0 else 0
            digit2 = int(num2[j]) if j >= 0 else 0
            
            total = digit1 + digit2 + carry
            carry = total // 10
            result.append(str(total % 10))
            
            i -= 1
            j -= 1
        
        return ''.join(reversed(result))


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    num1_1 = "11"
    num2_1 = "123"
    result1 = solution.addStrings(num1_1, num2_1)
    print(f"输入: num1 = \"{num1_1}\", num2 = \"{num2_1}\"")
    print(f"输出: \"{result1}\"")
    print()
    
    # 测试用例2
    num1_2 = "456"
    num2_2 = "77"
    result2 = solution.addStrings(num1_2, num2_2)
    print(f"输入: num1 = \"{num1_2}\", num2 = \"{num2_2}\"")
    print(f"输出: \"{result2}\"")
