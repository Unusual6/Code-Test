"""
83. 删除排序链表中的重复元素
难度：简单
给定一个已排序的链表的头 head，删除所有重复的元素，使每个元素只出现一次。
示例：输入：head = [1,1,2]  输出：[1,2]
"""
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        current = head
        while current and current.next:
            if current.val == current.next.val:
                current.next = current.next.next
            else:
                current = current.next
        return head
