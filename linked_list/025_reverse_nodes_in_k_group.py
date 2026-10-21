"""
25. K 个一组翻转链表 (Reverse Nodes in k-Group)

难度：困难

题目描述：
给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。
k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

示例 1：
输入：head = [1,2,3,4,5], k = 2
输出：[2,1,4,3,5]

示例 2：
输入：head = [1,2,3,4,5], k = 3
输出：[3,2,1,4,5]

提示：
链表中的节点数目为 n
1 <= k <= n <= 5000
0 <= Node.val <= 1000

进阶：你可以设计一个只用 O(1) 额外内存空间的算法解决此问题吗？
"""

from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        """
        迭代法：先计算长度，然后每k个一组翻转
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        # 计算链表长度
        length = 0
        current = head
        while current:
            length += 1
            current = current.next
        
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        
        # 每k个一组翻转
        while length >= k:
            # 翻转当前k个节点
            curr = prev.next
            for _ in range(k - 1):
                next_node = curr.next
                curr.next = next_node.next
                next_node.next = prev.next
                prev.next = next_node
            
            prev = curr
            length -= k
        
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
    k1 = 2
    result1 = solution.reverseKGroup(head1, k1)
    print(f"输入: head = [1,2,3,4,5], k = {k1}")
    print(f"输出: {print_linked_list(result1)}")
    print()
    
    # 测试用例2
    head2 = create_linked_list([1, 2, 3, 4, 5])
    k2 = 3
    result2 = solution.reverseKGroup(head2, k2)
    print(f"输入: head = [1,2,3,4,5], k = {k2}")
    print(f"输出: {print_linked_list(result2)}")
