"""
82. 删除排序链表中的重复元素 II
难度：中等
给定一个已排序的链表的头 head，删除原始链表中所有重复数字的节点，只留下不同的数字。
示例：输入：head = [1,2,3,3,4,4,5]  输出：[1,2,5]
"""
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        while prev.next:
            curr = prev.next
            while curr.next and curr.val == curr.next.val:
                curr = curr.next
            if prev.next == curr:
                prev = prev.next
            else:
                prev.next = curr.next
        return dummy.next
