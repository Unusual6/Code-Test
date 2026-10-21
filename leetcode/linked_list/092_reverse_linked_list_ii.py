"""
92. 反转链表 II
难度：中等
给你单链表的头指针 head 和两个整数 left 和 right，其中 left <= right。
请你反转从位置 left 到位置 right 的链表节点，返回反转后的链表。
示例：输入：head = [1,2,3,4,5], left = 2, right = 4  输出：[1,4,3,2,5]
"""
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        if not head or left == right:
            return head
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        for _ in range(left - 1):
            prev = prev.next
        curr = prev.next
        for _ in range(right - left):
            next_node = curr.next
            curr.next = next_node.next
            next_node.next = prev.next
            prev.next = next_node
        return dummy.next
