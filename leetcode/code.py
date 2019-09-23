from jzoffer.ListNode import ListNode, TreeLinkNode
from itertools import product


class Solution:
    # 1.two sum
    def twoSum(self, nums, target):
        keyindex = {}
        for i in range(len(nums)):
            if keyindex.get(nums[i]) is None:
                keyindex[nums[i]] = [i]
            else:
                keyindex[nums[i]].append(i)

        keyindex = sorted(keyindex.items(),
                          key=lambda x: x[0])  # 这时keyindex变成了一个list，每个元素是一个tuple，tuple的第一个元素是key,第二个元素是value
        indexs, nums = [], []
        for e in keyindex:
            indexs += e[1]
            nums += [e[0]] * len(e[1])
        point1, point2 = 0, len(nums) - 1
        while point1 < point2:
            sum = nums[point1] + nums[point2]
            if sum == target:
                return [indexs[point1], indexs[point2]]
            elif sum < target:
                point1 += 1
            else:
                point2 -= 1
        return []

    # 2、
    def addTwoNumbers(self, l1, l2):
        root = ListNode(0)
        l3 = root
        last = 0
        while l1 and l2:
            temp = l1.val + l2.val + last
            newtemp = temp % 10
            last = int(temp / 10)
            l3.next = ListNode(newtemp)
            l1 = l1.next
            l2 = l2.next
            l3 = l3.next

        if l1 is not None:
            while l1:
                temp = l1.val + last
                newtemp = temp % 10
                last = int(temp / 10)
                l3.next = ListNode(newtemp)
                l1 = l1.next
                l3 = l3.next


        else:
            while l2:
                temp = l2.val + last
                newtemp = temp % 10
                last = int(temp / 10)
                l3.next = ListNode(newtemp)
                l2 = l2.next
                l3 = l3.next
        if last != 0:
            l3.next = ListNode(last)
        return root.next

    # 3、Longest Substring Without Repeating Characters
    def lengthOfLongestSubstring(self, s):
        if len(s) <= 1: return len(s)
        maxlen = 0
        point1, point2 = 0, 1
        while point2 < len(s):
            if s[point2] in s[point1:point2]:
                maxlen = max(maxlen, point2 - point1)
                index = s[point1:point2].find(s[point2])
                point1 += index + 1
            else:
                point2 += 1

        maxlen = max(maxlen, point2 - point1)
        return maxlen

    # 4、Median of Two Sorted Arrays
    def findMedianSortedArrays(self, nums1, nums2):
        l = int((len(nums1) + len(nums2)) / 2) + 1
        newls = []
        i, j = 0, 0
        while i < len(nums1) and j < len(nums2) and len(newls) < l:
            if nums1[i] < nums2[j]:
                newls.append(nums1[i])
                i += 1
            else:
                newls.append(nums2[j])
                j += 1

        while j < len(nums2) and len(newls) < l:
            newls.append(nums2[j])
            j += 1

        while i < len(nums1) and len(newls) < l:
            newls.append(nums1[i])
            i += 1

        if (len(nums1) + len(nums2)) % 2 == 1:
            return newls[l - 1]
        else:
            return float(newls[l - 1] + newls[l - 2]) / 2

    # 5\Longest Palindromic Substring
    # 解释:由于回文字符有单核还有双核，所以，采用在每个字符后面都加上一个#,这样就可以将双核也转换成单核，不必再判别两次
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        news = []
        for c in s:
            news.append(c)
            news.append('#')
        s = news[:-1]
        maxstr = ''
        for i in range(len(s)):
            count = 1
            if s[i] == '#':
                count = 0
            p = i - 1
            q = i + 1
            while p >= 0 and q < len(s) and s[p] == s[q]:
                if s[p] != '#': count += 2
                p -= 1
                q += 1
            if len(maxstr) < count: maxstr = ''.join(s[p + 1:q]).replace('#', '')
        return maxstr

    # 6. ZigZag Conversion
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if numRows == 1:
            return s

        news = ''
        t = 2 * numRows - 2
        cols = int(len(s) / t) + 1
        for i in range(numRows):
            for j in range(cols + 1):
                mid = t * j + i
                pos = mid - i * 2
                if i >= 1 and i < numRows - 1 and pos >= 0 and pos < len(s): news += s[pos]
                if mid >= 0 and mid < len(s): news += s[mid]
        return news

    # 7. Reverse Integer
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x >= 0:
            sign = 1
        else:
            sign = -1
        if not str.isdigit(str(x)[0]): x = str(x)[1:]
        x = int(str(x)[::-1])
        x *= sign
        if x > 2147483648 or x < -2147483648: return 0
        return x

    # 8. String to Integer (atoi)
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        str = str.lstrip()
        if str == '': return 0
        sign = 1
        if str[0] == '-':
            sign = -1
            str = str[1:]
        elif str[0] == '+':
            str = str[1:]

        if str == '': return 0
        end = len(str)
        for i in range(len(str)):
            if not str[i].isdigit():
                end = i
                break
        str = str[:end]
        if not str.isdigit(): return 0
        num = sign * int(str)
        if num > 2147483647: num = 2147483648
        if num < -2147483648: num = -2147483648
        return num

    # 9. Palindrome Number
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0: return False
        return int(str(x)[::-1]) == x

    # 10. Regular Expression Matching
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        for i in range(m + 1):  # 这个地方从0开始是为了防止s为空串的情况
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    dp[i][j] = dp[i][j - 2] or (  # 情况1：s:b   p:a*b   这时，i保持不动，p的结果向前移动
                            i > 0 and dp[i - 1][j] and (
                            s[i - 1] == p[j - 2] or p[j - 2] == '.'))  # 情况2： s:aaab  p:a*b  这时，如果i=1,j=1,则保持j不动，将i向前移动

                else:
                    dp[i][j] = i > 0 and (p[j - 1] == s[i - 1] or p[j - 1] == '.') and dp[i - 1][j - 1]
        return dp[m][n]

    # 11. Container With Most Water:即计算图中的阴影部分的面积
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        p1 = 0
        p2 = len(height) - 1
        max = -1
        while p1 < p2:
            dis = (p2 - p1)
            temp = min(height[p1], height[p2]) * dis
            if max < temp: max = temp
            if height[p1] < height[p2]:
                p1 += 1
            else:
                p2 -= 1
        return max

    # 12. Integer to Roman
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        s = []
        base = 1000
        maps = {"1": "I", "5": "V", "10": "X", "50": "L", "100": "C", "500": "D", "1000": "M"}
        while num > 0:
            tp = int(num / base)
            num = num % base
            basestr = maps[str(int(base))]
            if tp < 4:
                s.append(basestr * tp)
            elif tp == 4:
                s.append(basestr + maps[str(int(base * 5))])
            elif tp < 9:
                s.append(maps[str(int(base * 5))] + basestr * (tp - 5))
            elif tp == 9:
                s.append(basestr + maps[str(int(base * 10))])
            base /= 10
        return ''.join(s)

    # 这个解法比上面的快很多，因为少了很多判断
    def intToRoman2(self, num):
        """
        :type num: int
        :rtype: str
        """
        s = ''
        basels = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        baseRoman = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
        for i in range(len(basels)):
            if basels[i] <= num:
                tp = int(num / basels[i])
                num = num % basels[i]
                s += baseRoman[i] * tp
            if num == 0: break
        return s

    #     13. Roman to Integer
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        basels = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        baseRoman = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
        num = 0
        point1 = 0
        point2 = 0
        while point1 < len(s):
            if s[point1] == baseRoman[point2]:
                num += basels[point2]
                point1 += 1
            elif s[point1:point1 + 2] == baseRoman[point2]:
                num += basels[point2]
                point1 += 2
            else:
                point2 += 1
        return num

    # 14. Longest Common Prefix
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if strs == None or len(strs) == 0:
            return ''

        s = strs[0]
        for i in range(1, len(strs)):
            j = 0
            lowindex = min(len(s), len(strs[i]))
            while j < lowindex:
                if s[j] != strs[i][j]: break
                j += 1
            s = s[:j]
        return s

    #     15. 3Sum：找出输入list中所有和为0的三元组
    # 思想:其实就是固定一个数，然后利用双指针寻找一个二元组
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums = sorted(nums)
        re = []
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]: continue
            j = i + 1
            k = len(nums) - 1
            while j < k:
                sum = nums[j] + nums[k]
                if sum < -nums[i]:
                    j += 1
                elif sum > -nums[i]:
                    k -= 1
                else:
                    re.append([nums[i], nums[k], nums[j]])
                    while j < k and nums[j + 1] == nums[j]:
                        j += 1
                    j += 1
                    while j < k and nums[k - 1] == nums[k]:
                        k -= 1
                    k -= 1
        return re

    # 16. 3Sum Closest
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums = sorted(nums)
        mindis = 1000000000
        minsum = 1000000000
        for i in range(len(nums) - 2):
            j = i + 1
            k = len(nums) - 1
            while j < k:
                sum = nums[i] + nums[j] + nums[k]
                dis = abs(sum - target)
                if dis < mindis:
                    mindis = dis
                    minsum = sum
                if dis == 0:
                    return minsum

                if sum - target > 0:
                    k -= 1
                else:
                    j += 1
        return minsum

    num_dict = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"], ["j", "k", "l"], ["m", "n", "o"],
                ["p", "q", "r", "s"], ["t", "u", "v"], ["w", "x", "y", "z"]]
    mapls = []

    # 17. Letter Combinations of a Phone Number
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if digits == None or len(digits) == 0: return []
        if len(digits) == 1: return self.num_dict[int(digits[0]) - 2]
        if digits[0] == '1':
            self.letterCombinations(digits[1:])
        else:
            re = self.letterCombinations(digits[1:])
            mapNum = self.num_dict[int(digits[0]) - 2]
            return [n + r for r, n in product(re, mapNum)]
            # print(digits,re)
            # for r,n in product(re,mapNum):
            #     self.mapls.append(n+r)

        # newmapls = self.mapls.copy()
        # self.mapls.clear() #需要clear，不然可能多次调用后所有的结果都放在了maols中
        # return newmapls

    def letterCombinations2(self, digits):
        num_dict = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"], ["j", "k", "l"], ["m", "n", "o"],
                    ["p", "q", "r", "s"], ["t", "u", "v"], ["w", "x", "y", "z"]]
        if digits == None or len(digits) == 0: return []
        if len(digits) == 1: return num_dict[int(digits[0]) - 2]
        mapls = num_dict[int(digits[len(digits) - 1]) - 2]
        for i in reversed(range(len(digits) - 1)):
            mapNum = num_dict[int(digits[i]) - 2]
            newmaps = []
            for n, pre in product(mapNum, mapls):
                newmaps.append(n + pre)
            mapls = newmaps
        return mapls

    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        # if len(nums) == 4:
        #     if nums[0]+nums[1]+nums[2]+nums[3] == target: return nums
        #     return []

        nums = sorted(nums)
        rels = []
        for i in range(len(nums) - 3):
            for j in range(i + 1, len(nums) - 2):
                sum = nums[i] + nums[j] - target
                start = j + 1
                end = len(nums) - 1
                while start < end:
                    temp = sum + nums[start] + nums[end]
                    if temp > 0:
                        while end > 0 and nums[end] == nums[end - 1]:
                            end -= 1
                        end -= 1
                    elif temp < 0:
                        while start < end and nums[start + 1] == nums[start]:
                            start += 1
                        start += 1
                    else:
                        if [nums[i], nums[j], nums[start], nums[end]] not in rels:
                            rels.append([nums[i], nums[j], nums[start], nums[end]])
                        end -= 1
                        start += 1

        return rels

    #     19. Remove Nth Node From End of List
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        nodels = []
        point = head
        while point:
            nodels.append(point)
            point = point.next

        if n > len(nodels): return None
        if n == len(nodels): return head.next
        pre = nodels[len(nodels) - n - 1]
        pre.next = pre.next.next
        return head

    def removeNthFromEnd2(self, head, n):
        pre = head
        point = head
        count = 0
        while point:
            count += 1
            if count > n + 1:
                pre = pre.next
            point = point.next
        if count == n: return head.next
        pre.next = pre.next.next
        return head

    #     20. Valid Parentheses
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        maps = []
        dict = {']': '[', '}': '{', ')': '('}
        for c in s:
            if c == '[' or c == '{' or c == '(':
                maps.append(c)
            else:
                if c == ']' or c == '}' or c == ')':
                    if len(maps) > 0:
                        ch = maps.pop()
                        if ch == dict[c]:
                            pass
                        else:
                            return False
                    else:
                        return False

        if len(maps) != 0:
            return False
        return True

    # 21. Merge Two Sorted Lists
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        point1 = l1
        point2 = l2
        newls = ListNode(0)
        point3 = newls
        while point1 and point2:
            if point1.val < point2.val:
                point3.next = point1

                point1 = point1.next
            else:
                point3.next = point2

                point2 = point2.next
            point3 = point3.next

        if point1: point3.next = point1

        if point2: point3.next = point2

        return newls.next

    # 22. Generate Parentheses：采用DFS的思想   **********************
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        if n == 1: return ["()"]
        mapls = []
        self.parenthesisDFS(n, n, "", mapls)
        return list(set(mapls))

    # 23. Merge k Sorted Lists
    def parenthesisDFS(self, left, right, s, mapls):
        if left == 0 and right == 0:
            mapls.append(s)
            return
        if left < 0 or right < 0 or right < left: return
        self.parenthesisDFS(left - 1, right, s + "(", mapls)
        self.parenthesisDFS(left, right - 1, s + ")", mapls)

    def mergeKLists(self, lists: 'List[ListNode]') -> 'ListNode':
        if lists is None or len(lists) == 0: return None
        if len(lists) == 1: return lists[0]
        mid = int(len(lists) / 2)
        left = self.mergeKLists(lists[:mid])
        right = self.mergeKLists(lists[mid:])
        proot = ListNode(0)
        temp = proot
        p, q = left, right
        while p and q:
            if p.val < q.val:
                temp.next = p
                p = p.next
            else:
                temp.next = q
                q = q.next
            temp = temp.next

        if p: temp.next = p
        if q: temp.next = q
        return proot.next

    # 24. Swap Nodes in Pairs
    def swapPairs(self, head: ListNode) -> ListNode:
        newroot = ListNode(0)
        newroot.next = head
        pre = newroot
        p = head
        while p and p.next:
            temp1 = p.next
            temp2 = p.next.next
            temp1.next = p
            p.next = temp2
            pre.next = temp1
            pre = p
            p = p.next
        return newroot.next

    # 25. Reverse Nodes in k-Group:要考虑length<k的情况
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        if head is None: return
        if k == 1: return head
        root = ListNode(0)
        point1 = head
        point2 = head
        count = 1
        pre = root
        pre.next = head
        next = None
        while point2:
            if count % k == 0:
                next = point2.next
                point2.next = None
                head, tail = self.reverseLink(point1)
                pre.next = head
                tail.next = next
                point1 = next
                pre = tail
                point2 = point1
                count += 1
            if point2:
                point2 = point2.next
                count += 1

        return root.next

    # 链表反转：就是一个结点一个结点的转向
    def reverseLink(self, head: ListNode) -> ListNode:
        pre, next = None, None
        tail = head
        while head:
            next = head.next
            head.next = pre
            pre = head
            head = next
        return pre, tail

#     109. Convert Sorted List to Binary Search Tree
    #这种解法其实就是模拟二叉平衡树的中序遍历生成有序序列的过程，比较巧妙，时间复杂度为O(N)，空间复杂度为O(logN)
    def GetSize(self, head):
        count = 0
        p = head
        while p:
            count += 1
            p = p.next
        return count

    def sortedListToBST(self, head: ListNode):
        if head is None: return None
        size = self.GetSize(head)

        def toMidVisit(l, r):
            nonlocal head
            if l > r: return None
            mid = (l + r) // 2
            left = toMidVisit(l, mid - 1)
            node = TreeLinkNode(head.val)
            node.left = left
            head = head.next
            right = toMidVisit(mid + 1, r)
            node.right = right
            return node

        return toMidVisit(0, size - 1)
    # 113. Path Sum II
    def pathSum(self, root: TreeLinkNode, sum: int):
        if root is None: return []
        allpaths = []

        def breadthFirst(root, sum, papath):
            nonlocal allpaths
            papath.append(root.val)
            if root.left == None and root.right == None and root.val == sum:
                allpaths.append(papath)
            else:
                sum -= root.val
                if sum <= 0: return None
                temp = [i for i in papath]
                if root.left: breadthFirst(root.left, sum, papath)
                if root.right: breadthFirst(root.right, sum, temp)

        initpath = []
        breadthFirst(root, sum, initpath)
        return allpaths

    # 59. Spiral Matrix II
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        list = []
        for i in range(n):
            list.append([-1 for _ in range(n)])
        direct = 1
        count = 0
        i, j = 0, 0
        while count < pow(n, 2):
            if direct == 1:
                while 0<= i <n and 0<= j < n and list[i][j] == -1:
                    list[i][j] = count + 1
                    count += 1
                    if j == n - 1 or list[i][j+1] != -1:
                        direct = 2
                        i += 1
                        break
                    else:
                        j += 1
            if direct == 2:
                while 0<= i <n and 0<= j < n and list[i][j] == -1:
                    list[i][j] = count + 1
                    count += 1
                    if i == n - 1  or list[i+1][j] != -1:
                        direct = 3
                        j -= 1
                        break
                    else:
                        i += 1
            if direct == 3:
                while 0<= i <n and 0<= j < n and list[i][j] == -1:
                    list[i][j] = count + 1
                    count += 1
                    if j == 0  or list[i][j-1] != -1:
                        direct = 4
                        i -= 1
                        break
                    else:
                        j -= 1
            if direct == 4:
                while 0<= i <n and 0<= j < n and list[i][j] == -1:
                    list[i][j] = count + 1
                    count += 1
                    if i == 0 or list[i-1][j] != -1:
                        direct = 1
                        j += 1
                        break
                    else:
                        i -= 1
        return list


    def generateMatrix2(self, n):
        A, lo = [[n * n]], n * n
        while lo > 1:
            lo, hi = lo - len(A), lo
            A = [[i for i in range(lo, hi)]]+ list(zip(*A[::-1]))
        return A

    # 915. Partition Array into Disjoint Intervals
    def partitionDisjoint(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        point = 0
        while point < len(A):
            left_i = 0
            right_i =point + 1
            left_max_index = left_i
            right_min_index = right_i
            while left_i < point + 1:
                  if A[left_i] >= A[left_max_index]:
                      left_max_index = left_i
                  left_i += 1
            while right_i < len(A):
                if A[right_i] <= A[right_min_index]:
                    right_min_index = right_i
                right_i += 1
            if A[left_max_index] <= A[right_min_index]:
                return point + 1
            else:
                point = right_min_index
        return 0

#31. Next Permutation

    def exchange(self,nums):
        return nums[::-1]

    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """

        for i in range(len(nums) - 2, -1, -1):
            if nums[i] >= nums[-1]:
                temp = nums[i]
                nums.append(temp)
                nums = list(nums[:i]) + list(nums[i+1:])
                # index = i + 1
                # while index < len(nums):
                #     nums[index - 1] = nums[index]
                #     index += 1
                # nums[-1] = temp
            else:
                index = i + 1

                while index < len(nums):
                    if nums[index] > nums[i]:
                        temp = nums[i]
                        nums[i] = nums[index]
                        nums[index] = temp
                        break
                    index += 1
                break
        print(nums)

# 33. Search in Rotated Sorted Array
    def search(self, nums, target):
        if nums is None or len(nums) == 0: return -1
        left = 0
        right = len(nums) - 1
        if nums[0] == target: return 0
        while left < right:
            mid = int((left + right)/2)
            if nums[mid] == target:
                return mid
            if nums[right] == target:
                return right
            if nums[left] == target:
                return left
            if nums[mid] > nums[left]:
                if nums[left] < target < nums[mid]:
                    left = left
                    right = mid
                else:
                    left = mid + 1
                    right = right - 1
            else:
                if nums[mid] < target < nums[right]:
                    left = mid + 1
                    right = right - 1
                else:
                    left = left + 1
                    right = mid - 1
        if right == left and nums[right] == target:return right
        return -1

#34. Find First and Last Position of Element in Sorted Array
    def searchRange(self, nums, target):
        pass


# 98. Validate Binary Search Tree
    def isValidBST(self, root) -> bool:
        pre = -10000
        T = root
        stack = []
        while T or len(stack) > 0:
            while(not T):
                T = T.left
                if T:
                    stack.push(T)

            if len(stack) > 0:
                T = stack.pop()
                print(T.val)
                if pre > T.val:
                    return False
                T = T.right
        return True

# 881. Boats to Save People:存在疑问
    def numRescueBoats(self,people, limit):
        if max(people) > limit: return 0
        people = sorted(people,reverse=True)
        count, left, right = 0, 0, len(people) - 1
        while left <= right:
            if(people[left] + people[right] <= limit):
                count += 1
                left += 1
                right -= 1
            else:
                left += 1
                count += 1
        return count

#     886. Possible Bipartition
    def possibleBipartition(self, N, dislikes):
        """
        :type N: int
        :type dislikes: List[List[int]]
        :rtype: bool
        """
        graph = {}
        for i in range(1,N+1):
            graph[i] = []
        for u, v in dislikes:
            if u not in graph.keys():
                graph[u] = []
            if v not in graph.keys():
                graph[v] = []
            graph[u].append(v)
            graph[v].append(u)

        def dfs(node, c=-1):
            if node in color.keys():
                return color[node] == c
            color[node] = c
            for n in graph[node]:
                if not dfs(n, ~c):
                    return False
            return True

        color = {}
        for node in range(1,N+1):
            if node not in color.keys():
                if dfs(node,-1) == False:
                    return False
        return True

    def fairCandySwap(self, A, B):
        Sa = sum(A)
        Sb = sum(B)
        B = set(B)
        for a in A:
            if a + (Sb - Sa)/ 2 in B:
                return [a, int(a+(Sb - Sa)/2)]
        return None

#     889. Construct Binary Tree from Preorder and Postorder Traversal
    def constructFromPrePost(self, pre, post):
        """
        :type pre: List[int]
        :type post: List[int]
        :rtype: TreeNode
        """
        if pre is None or post is None or len(pre) == 0 or len(post) == 0:
            return None
        pre_idx = 1
        post_idx = 0
        root = TreeLinkNode(pre[0])
        stack = (root,None)
        while stack and pre_idx < len(pre):
            selectedNode = stack[0]
            newNode = TreeLinkNode(pre[pre_idx])
            if not stack[0].left:
                stack[0].left = newNode
            elif not stack[0].right:
                stack[0].right = newNode
            else:
                return None
            stack = (newNode,stack)

            while stack and post[post_idx] == stack[0].val:
                stack = stack[1]
                post_idx += 1
            pre_idx += 1
        return root

    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        count = []
        for i, row in enumerate(board):
            for j, c in enumerate(row):
                if c != '.':
                    if (c,i) in count:
                        pass
                    if (c,j) in count:
                        pass
                    if (c,int(i/3),int(j/3)) in count:
                        pass
                    count += [(c,i),(c,j),(c,int(i/3),int(j/3))]
        return len(count) == len(set(count))

#402. Remove K Digits
    def removeKdigits(self, num, k):
        """
        :type num: str
        :type k: int
        :rtype: str
        """
        if k >= len(num):
            return "0"

        flags = [0 for _ in range(len(num))]
        flags[-1] = int(num[-1])
        for i in reversed(range(len(num)-1)):
            temp = int(num[i]) - int(num[i + 1])
            if temp == 0:
                flags[i] = flags[i + 1]
            else:
                flags[i] = temp

        count = 0
        newstr = ''
        for i in range(len(num)):
            if count >= k:
                newstr += num[i:]
                break
            if flags[i] > 0:
                count += 1
            else:
                newstr += num[i]
        if len(num) - len(newstr) < k:
            return self.removeKdigits(newstr, k - len(num) + len(newstr))
        return str(int(newstr))

    def removeKdigits2(self, num, k):
        if k >= len(num):
            return "0"

        stack = []
        for c in num:
            while k > 0 and len(stack) > 0 and stack[-1] > c:
                stack.pop()
                k -= 1
            stack.append(c)

        while k > 0 and len(stack) > 0:
            stack.pop()
            k -= 1

        return str(int(str(''.join(stack))))

#300
    def lengthOfLIS(self, nums):
        if nums is None or len(nums) <= 0:
            return 0
        res = [nums[0]]
        for n in nums[1:]:
            if n > res[-1]:
                res.append(n)
            else:
                low = 0
                high = len(res)
                while low < high:
                    mid = int((high - low) / 2) + low
                    if res[mid] < n:
                        low = mid + 1
                    elif res[mid] == n:
                        high = mid
                        break
                    else:
                        high = mid
                res[high] = n
        return len(res)













s = Solution()
# input = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
# t = s.isValidSudoku(input)
# print(t)

# res = s.removeKdigits2("1234567890",7)
# print(res)
nums = [3,5,6,2,5,4,5,19,5,6,7,12]
t = s.lengthOfLIS(nums)
print(t)




















































