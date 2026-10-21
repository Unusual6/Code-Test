"""
143. 重排链表
难度：中等
给定一个单链表 L 的头节点 head，将其重新排列为：L0 → Ln → L1 → Ln-1 → ...
示例：输入：head = [1,2,3,4]  输出：[1,4,2,3]
"""
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        if not head or not head.next:
            return
        slow, fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        prev = None
        curr = slow.next
        slow.next = None
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        first, second = head, prev
        while second:
            f_next = first.next
            s_next = second.next
            first.next = second
            second.next = f_next
            first = f_next
            second = s_next
