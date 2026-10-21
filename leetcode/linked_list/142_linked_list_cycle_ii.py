"""
142. 环形链表 II (Linked List Cycle II)

难度：中等

题目描述：
给定一个链表的头节点 head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null 。
如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。
为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。
如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。
不允许修改 链表。

示例 1：
输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。

示例 2：
输入：head = [1,2], pos = 0
输出：返回索引为 0 的链表节点
解释：链表中有一个环，其尾部连接到第一个节点。

示例 3：
输入：head = [1], pos = -1
输出：返回 null
解释：链表中没有环。

提示：
链表中节点的数目范围在范围 [0, 10^4] 内
-10^5 <= Node.val <= 10^5
pos 的值为 -1 或者链表中的一个有效索引

进阶：你是否可以使用 O(1) 空间解决此题？
"""

from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        快慢指针法：先判断是否有环，然后找入环节点
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not head or not head.next:
            return None
        
        # 第一步：判断是否有环
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                # 有环，找入环节点
                # 将一个指针重新指向head，两个指针每次走一步
                slow = head
                while slow != fast:
                    slow = slow.next
                    fast = fast.next
                return slow
        
        return None


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    def create_linked_list_with_cycle(nums, pos):
        dummy = ListNode(0)
        current = dummy
        cycle_node = None
        nodes = []
        
        for i, num in enumerate(nums):
            current.next = ListNode(num)
            current = current.next
            nodes.append(current)
            if i == pos:
                cycle_node = current
        
        if pos >= 0:
            current.next = cycle_node
        
        return dummy.next, nodes, cycle_node
    
    # 测试用例1
    head1, nodes1, cycle_node1 = create_linked_list_with_cycle([3, 2, 0, -4], 1)
    result1 = solution.detectCycle(head1)
    print(f"输入: head = [3,2,0,-4], pos = 1")
    print(f"输出: 入环节点值为 {result1.val if result1 else 'null'}")
    print()
    
    # 测试用例2
    head2, nodes2, cycle_node2 = create_linked_list_with_cycle([1], -1)
    result2 = solution.detectCycle(head2)
    print(f"输入: head = [1], pos = -1")
    print(f"输出: {result2}")
