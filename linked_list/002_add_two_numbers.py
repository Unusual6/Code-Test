"""
2. 两数相加 (Add Two Numbers)

难度：中等

题目描述：
给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，
并且每个节点只能存储 一位 数字。
请你将两个数相加，并以相同形式返回一个表示和的链表。
你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

示例 1：
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.

示例 2：
输入：l1 = [0], l2 = [0]
输出：[0]

示例 3：
输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
输出：[8,9,9,9,0,0,0,1]

提示：
每个链表中的节点数在范围 [1, 100] 内
0 <= Node.val <= 9
题目数据保证列表表示的数字不含前导零
"""

from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        模拟竖式加法：逐位相加，处理进位
        时间复杂度：O(max(n1, n2))
        空间复杂度：O(1)
        """
        dummy = ListNode(0)
        current = dummy
        carry = 0
        
        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            
            total = val1 + val2 + carry
            carry = total // 10
            
            current.next = ListNode(total % 10)
            current = current.next
            
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        
        return dummy.next


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    def create_linked_list(nums):
        dummy = ListNode(0)
        current = dummy
        for num in nums:
            current.next = ListNode(num)
            current = current.next
        return dummy.next
    
    def print_linked_list(head):
        result = []
        current = head
        while current:
            result.append(current.val)
            current = current.next
        return result
    
    # 测试用例1
    l1 = create_linked_list([2, 4, 3])
    l2 = create_linked_list([5, 6, 4])
    result = solution.addTwoNumbers(l1, l2)
    print(f"输入: l1 = [2,4,3], l2 = [5,6,4]")
    print(f"输出: {print_linked_list(result)}")
    print()
    
    # 测试用例2
    l3 = create_linked_list([0])
    l4 = create_linked_list([0])
    result2 = solution.addTwoNumbers(l3, l4)
    print(f"输入: l1 = [0], l2 = [0]")
    print(f"输出: {print_linked_list(result2)}")
