"""
138. 复制带随机指针的链表
难度：中等
给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random。构造这个链表的深拷贝。
示例：输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]  输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
"""

class Node:
    def __init__(self, x: int, next=None, random=None):
        self.val = int(x)
        self.next = next
        self.random = random

class Solution:
    def copyRandomList(self, head):
        if not head:
            return None
        curr = head
        while curr:
            clone = Node(curr.val, curr.next)
            curr.next = clone
            curr = clone.next
        curr = head
        while curr:
            if curr.random:
                curr.next.random = curr.random.next
            curr = curr.next.next
        curr = head
        clone_head = head.next
        clone = clone_head
        while clone.next:
            curr.next = curr.next.next
            clone.next = clone.next.next
            curr = curr.next
            clone = clone.next
        curr.next = None
        return clone_head
