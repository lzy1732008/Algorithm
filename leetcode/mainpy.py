from leetcode.code import Solution
from leetcode.DP import minPathSum, minimumTotal
from jzoffer.ListNode import ListNode

if __name__=='__main__':
    # l1 = ListNode(-10)
    # node1 = ListNode(-3)
    # node2 = ListNode(0)
    # node3 = ListNode(5)
    # node4 = ListNode(9)
    #
    # l1.next = node1
    # node1.next = node2
    # node2.next = node3
    # node3.next = node4
    # node4.next = None
    #
    # l2 = ListNode(1)
    # node1 = ListNode(6)
    # node2 = ListNode(7)
    # l2.next = node1
    # node1.next = node2
    #
    # l3 = ListNode(4)
    # node1 = ListNode(5)
    # node2 = ListNode(7)
    # l3.next = node1
    # node1.next = node2


    s = Solution()
    t = s.search([4,5,6,0,1,2],-1)
    print(t)
    #
    # while t:
    #     print(t.val)
    #     t = t.next


    # nums =  [3,4,5,-7,1,2]
    # t = s.maxSubArray2(nums)
    # print(t)

#     grid = [
#      [2],
#     [3,4],
#    [6,5,7],
#   [4,1,8,3]
# ]
#     print(minimumTotal(grid))


