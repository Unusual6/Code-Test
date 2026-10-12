"""
234. 回文链表
难度：简单
给你一个单链表的头节点 head，请你判断该链表是否为回文链表。
示例：输入：head = [1,2,2,1]  输出：true
"""
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return True
        slow, fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        prev = None
        curr = slow.next
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        first, second = head, prev
        while second:
            if first.val != second.val:
                return False
            first = first.next
            second = second.next
        return True
