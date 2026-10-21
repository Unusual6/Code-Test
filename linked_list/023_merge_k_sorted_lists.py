"""
23. 合并K个升序链表
难度：困难
给你一个链表数组，每个链表都已经按升序排列。请你将所有链表合并到一个升序链表中。
示例：输入：lists = [[1,4,5],[1,3,4],[2,6]]  输出：[1,1,2,3,4,4,5,6]
"""
from typing import Optional, List
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        heap = []
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(heap, (lst.val, i, lst))
        dummy = ListNode(0)
        current = dummy
        while heap:
            val, i, node = heapq.heappop(heap)
            current.next = node
            current = current.next
            if node.next:
                heapq.heappush(heap, (node.next.val, i, node.next))
        return dummy.next
