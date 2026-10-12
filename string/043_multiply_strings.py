"""
43. 字符串相乘 (Multiply Strings)

难度：中等

题目描述：
给定两个以字符串形式表示的非负整数 num1 和 num2，返回它们的乘积，它们的乘积也表示为字符串形式。
注意： 不能使用任何内置的 BigInteger 库或直接将输入转换为整数。

示例 1:
输入: num1 = "2", num2 = "3"
输出: "6"

示例 2:
输入: num1 = "123", num2 = "456"
输出: "56088"

提示:
1 <= num1.length, num2.length <= 200
num1 和 num2 仅由数字组成
num1 和 num2 都不含任何前导零，除了数字0本身。
"""


class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        """
        模拟竖式乘法：使用数组存储结果
        时间复杂度：O(n1 * n2)
        空间复杂度：O(n1 + n2)
        """
        if num1 == "0" or num2 == "0":
            return "0"
        
        m, n = len(num1), len(num2)
        result = [0] * (m + n)
        
        # 从后往前逐位相乘
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                mul = int(num1[i]) * int(num2[j])
                
                # 位置索引
                p1, p2 = i + j, i + j + 1
                
                # 加上之前的进位
                total = mul + result[p2]
                
                result[p2] = total % 10
                result[p1] += total // 10
        
        # 转换为字符串
        result_str = ''.join(map(str, result))
        return result_str.lstrip('0') or '0'


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    num1_1 = "2"
    num2_1 = "3"
    result1 = solution.multiply(num1_1, num2_1)
    print(f"输入: num1 = \"{num1_1}\", num2 = \"{num2_1}\"")
    print(f"输出: \"{result1}\"")
    print()
    
    # 测试用例2
    num1_2 = "123"
    num2_2 = "456"
    result2 = solution.multiply(num1_2, num2_2)
    print(f"输入: num1 = \"{num1_2}\", num2 = \"{num2_2}\"")
    print(f"输出: \"{result2}\"")
