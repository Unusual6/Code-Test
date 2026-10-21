"""
19. 删除链表的倒数第 N 个结点 (Remove Nth Node From End of List)

难度：中等

题目描述：
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

示例 1：
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]

示例 2：
输入：head = [1], n = 1
输出：[]

示例 3：
输入：head = [1,2], n = 1
输出：[1]

提示：
链表中结点的数目为 sz
1 <= sz <= 30
0 <= Node.val <= 100
1 <= n <= sz

进阶：你能尝试使用一趟扫描实现吗？
"""

from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        双指针法：快指针先走n步，然后快慢指针一起走
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        dummy = ListNode(0)
        dummy.next = head
        
        fast = dummy
        slow = dummy
        
        # 快指针先走n+1步
        for _ in range(n + 1):
            fast = fast.next
        
        # 快慢指针一起走
        while fast:
            fast = fast.next
            slow = slow.next
        
        # 删除倒数第n个节点
        slow.next = slow.next.next
        
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
    head1 = create_linked_list([1, 2, 3, 4, 5])
    n1 = 2
    result1 = solution.removeNthFromEnd(head1, n1)
    print(f"输入: head = [1,2,3,4,5], n = {n1}")
    print(f"输出: {print_linked_list(result1)}")
    print()
    
    # 测试用例2
    head2 = create_linked_list([1])
    n2 = 1
    result2 = solution.removeNthFromEnd(head2, n2)
    print(f"输入: head = [1], n = {n2}")
    print(f"输出: {print_linked_list(result2)}")
