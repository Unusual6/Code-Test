"""
206. 反转链表 (Reverse Linked List)

难度：简单

题目描述：
给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。

示例 1：
输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]

示例 2：
输入：head = [1,2]
输出：[2,1]

示例 3：
输入：head = []
输出：[]

提示：
链表中节点的数目范围是 [0, 5000]
-5000 <= Node.val <= 5000

进阶：链表可以选用迭代或递归方式完成反转。你能否用两种方法解决这道题？
"""

from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        迭代法：使用三个指针逐个反转
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        prev = None
        current = head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        return prev
    
    def reverseListRecursive(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        递归法
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not head or not head.next:
            return head
        
        new_head = self.reverseListRecursive(head.next)
        head.next.next = head
        head.next = None
        
        return new_head


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
    head1 = create_linked_list([1, 2, 3, 4, 5])
    result1 = solution.reverseList(head1)
    print(f"输入: head = [1,2,3,4,5]")
    print(f"输出: {print_linked_list(result1)}")
    print()
    
    # 测试用例2
    head2 = create_linked_list([1, 2])
    result2 = solution.reverseList(head2)
    print(f"输入: head = [1,2]")
    print(f"输出: {print_linked_list(result2)}")
