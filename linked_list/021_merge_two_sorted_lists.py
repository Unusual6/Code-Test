"""
21. 合并两个有序链表 (Merge Two Sorted Lists)

难度：简单

题目描述：
将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

示例 1：
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]

示例 2：
输入：l1 = [], l2 = []
输出：[]

示例 3：
输入：l1 = [], l2 = [0]
输出：[0]

提示：
两个链表的节点数目范围是 [0, 50]
-100 <= Node.val <= 100
l1 和 l2 均按 非递减顺序 排列
"""

from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """
        迭代法：使用dummy节点，逐个比较
        时间复杂度：O(n + m)
        空间复杂度：O(1)
        """
        dummy = ListNode(0)
        current = dummy
        
        while list1 and list2:
            if list1.val <= list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next
        
        # 连接剩余部分
        current.next = list1 if list1 else list2
        
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
    l1 = create_linked_list([1, 2, 4])
    l2 = create_linked_list([1, 3, 4])
    result1 = solution.mergeTwoLists(l1, l2)
    print(f"输入: l1 = [1,2,4], l2 = [1,3,4]")
    print(f"输出: {print_linked_list(result1)}")
    print()
    
    # 测试用例2
    l3 = create_linked_list([])
    l4 = create_linked_list([])
    result2 = solution.mergeTwoLists(l3, l4)
    print(f"输入: l1 = [], l2 = []")
    print(f"输出: {print_linked_list(result2)}")
