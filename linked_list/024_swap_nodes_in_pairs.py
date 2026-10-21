"""
24. 两两交换链表中的节点 (Swap Nodes in Pairs)

难度：中等

题目描述：
给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。
你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

示例 1：
输入：head = [1,2,3,4]
输出：[2,1,4,3]

示例 2：
输入：head = []
输出：[]

示例 3：
输入：head = [1]
输出：[1]

提示：
链表中节点的数目在范围 [0, 100] 内
0 <= Node.val <= 100
"""

from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        迭代法：使用dummy节点，每次交换一对节点
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        dummy = ListNode(0)
        dummy.next = head
        current = dummy
        
        while current.next and current.next.next:
            # 要交换的两个节点
            first = current.next
            second = current.next.next
            
            # 交换
            first.next = second.next
            second.next = first
            current.next = second
            
            # 移动到下一对
            current = first
        
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
    head1 = create_linked_list([1, 2, 3, 4])
    result1 = solution.swapPairs(head1)
    print(f"输入: head = [1,2,3,4]")
    print(f"输出: {print_linked_list(result1)}")
    print()
    
    # 测试用例2
    head2 = create_linked_list([])
    result2 = solution.swapPairs(head2)
    print(f"输入: head = []")
    print(f"输出: {print_linked_list(result2)}")
