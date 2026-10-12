"""
160. 相交链表 (Intersection of Two Linked Lists)

难度：简单

题目描述：
给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。
图示两个链表在节点 c1 开始相交：
题目数据 保证 整个链式结构中不存在环。
注意，函数返回结果后，链表必须 保持其原始结构 。

示例 1：
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
输出：Intersected at '8'
解释：相交节点的值为 8 。

示例 2：
输入：intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出：Intersected at '2'

示例 3：
输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
输出：No intersection

提示：
listA 中节点数目为 m
listB 中节点数目为 n
1 <= m, n <= 3 * 10^4
1 <= Node.val <= 10^5
0 <= skipA <= m
0 <= skipB <= n
如果 listA 和 listB 没有交点，intersectVal 为 0
如果 listA 和 listB 有交点，intersectVal == listA[skipA] == listB[skipB]

进阶：你能否设计一个时间复杂度为 O(m + n) 、仅用 O(1) 内存的解决方案？
"""

from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """
        双指针法：两个指针分别遍历两个链表，到达末尾后切换到另一个链表
        时间复杂度：O(m + n)
        空间复杂度：O(1)
        """
        if not headA or not headB:
            return None
        
        pA, pB = headA, headB
        
        while pA != pB:
            pA = pA.next if pA else headB
            pB = pB.next if pB else headA
        
        return pA


# 测试用例
if __name__ == "__main__":
    solution = Solution()
    
    def create_intersecting_lists(listA_vals, listB_vals, intersect_val, skipA, skipB):
        # 创建相交部分
        intersect = None
        if intersect_val != 0:
            intersect = ListNode(intersect_val)
            curr = intersect
            # 添加剩余的相交节点
            for i in range(skipA + 1, len(listA_vals)):
                curr.next = ListNode(listA_vals[i])
                curr = curr.next
        
        # 创建链表A
        dummyA = ListNode(0)
        currA = dummyA
        for i in range(skipA):
            currA.next = ListNode(listA_vals[i])
            currA = currA.next
        if intersect:
            currA.next = intersect
        
        # 创建链表B
        dummyB = ListNode(0)
        currB = dummyB
        for i in range(skipB):
            currB.next = ListNode(listB_vals[i])
            currB = currB.next
        if intersect:
            currB.next = intersect
        
        return dummyA.next, dummyB.next
    
    # 测试用例1
    headA1, headB1 = create_intersecting_lists([4, 1, 8, 4, 5], [5, 6, 1, 8, 4, 5], 8, 2, 3)
    result1 = solution.getIntersectionNode(headA1, headB1)
    print(f"输入: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5]")
    print(f"输出: Intersected at '{result1.val if result1 else 'null'}'")
    print()
    
    # 测试用例2
    headA2, headB2 = create_intersecting_lists([2, 6, 4], [1, 5], 0, 3, 2)
    result2 = solution.getIntersectionNode(headA2, headB2)
    print(f"输入: intersectVal = 0, listA = [2,6,4], listB = [1,5]")
    print(f"输出: {result2}")
