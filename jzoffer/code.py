import numpy as np
from functools import reduce, cmp_to_key
import ctypes
import re
from jzoffer.ListNode import TreeLinkNode
import heapq as hp


class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        if tinput == None or k > len(tinput):
            return
        if k == len(tinput):
            return sorted(tinput)
        pre = tinput[:k]
        last = tinput[k:]
        for i in range(len(last)):
            for j in range(k):
                if last[i] < pre[j]:
                    temp = last[i]
                    last[i] = pre[j]
                    pre[j] = temp
        return sorted(pre)

    def FindGreatestSumOfSubArray(self, array):
        max_value = array[0]
        end_index = 0
        seq = []
        for i in range(len(array)):
            if i == 0:
                seq.append(array[0])
            else:
                s = array[i] + seq[i - 1]
                if s > array[i]:
                    seq.append(s)
                else:
                    seq.append(array[i])
            if seq[i] > max_value:
                max_value = seq[i]
                end_index = i

        start_index = end_index

        while start_index - 1 >= 0:
            if seq[start_index - 1] > 0:
                start_index = start_index - 1
            else:
                break

        if start_index == end_index:
            return np.sum(array[start_index])
        else:
            return seq[end_index]

    def NumberOf1Between1AndN_Solution(self, n):
        all_str = ''
        for i in range(1, n + 1):
            all_str += str(i)
        return all_str.count('1')

    # 把数组排成最小的数
    def cmp(self, a, b):
        c = str(a) + str(b)
        d = str(b) + str(b)
        if c < d:
            return -1
        else:
            return 1
        return 0

    def PrintMinNumber(self, numbers):
        numbers = sorted(numbers, key=cmp_to_key(self.z))
        return reduce(lambda x, y: str(x) + str(y), numbers)

    # 丑数

    def GetUglyNumber_Solution(self, index):
        if index <= 0:
            return
        num_list = [0, 1]
        count = index - 1
        base = 1
        while count > 0:
            base += 1
            num_2, num_3, num_5 = base / 2, base / 3, base / 5
            print(num_2, num_3, num_5)
            if num_2 in num_list or num_3 in num_list or num_5 in num_list:
                num_list.append(base)
                count -= 1
        return num_list[index]

    # 由于一个丑数*2,3,5之后的数肯定也是丑数，因此，设置三个指针分别表示接下来要被计算的丑数。
    def GetUglyNumber_Solution2(self, index):
        if index < 7:
            return index
        stack = []
        p2, p3, p5 = 0, 0, 0
        newnum = 1
        stack.append(newnum)
        while len(stack) < index:
            newnum = min(stack[p2] * 2, min(stack[p3] * 3, stack[p5] * 5))

            if stack[p2] * 2 == newnum: p2 += 1
            if stack[p3] * 3 == newnum: p3 += 1
            if stack[p5] * 5 == newnum: p5 += 1

            stack.append(newnum)

        return newnum

    # 第一次只出现一次的字符串
    def FirstNotRepeatingChar(self, s):
        return s.index(list(filter(lambda c: s.count(c) == 1, s))[0]) if s else -1

    # 数组中的逆序对:使用归并排序的思想
    cnt = 0

    def InversePairs(self, data):
        if data:
            self.MergeSort(data, 0, len(data) - 1)
        return self.cnt

    def MergeSort(self, data, start, end):
        if start < end:
            mid = int((end + start) / 2)
            self.MergeSort(data, start, mid)
            self.MergeSort(data, mid + 1, end)
            self.Merge(data, start, mid, end)

    def Merge(self, data, start, mid, end):
        temp = []
        i, j = start, mid + 1
        while i <= mid and j <= end:
            if data[i] < data[j]:
                temp.append(data[i])
                i += 1
            else:
                temp.append(data[j])
                j += 1
                self.cnt += mid - i + 1

        while i <= mid:
            temp.append(data[i])
            i += 1

        while j <= end:
            temp.append(data[j])
            j += 1

        i = 0
        while i < len(temp):
            data[start + i] = temp[i]
            i += 1

    # 扑克牌顺子：大小王用0表示，可以模拟任何数字。
    # 如果是0,0,a,b,c的情景
    # 如果是0,a,b,c,d的情景
    # 0,0,0,0,a
    # 0,0,0,a
    # []
    def IsContinuous(self, numbers):
        if len(numbers) == 0:
            return False
        numbers = sorted(numbers)
        mis = 0;
        zeroCount = 0
        for i in range(len(numbers) - 1):
            if numbers[i] == 0:
                zeroCount += 1
            else:
                t = numbers[i + 1] - numbers[i]
                if t > 1:
                    mis += t - 1
                elif t == 0:
                    return False
        return mis <= zeroCount  # 注意这边是<=,因为0有可能个数很多，多余需要填补的

    # 孩子们的游戏：圆圈中最后剩下的数
    # 思想是：当只剩下最后一个人时，他的编号是0，那么就可以知道剩下两个人的时候被删除的是哪个，即(f(1)+m)%2
    # 这个递归出现了"maximum recursion depth exceeded"这个错误
    def LastRemaining_Solution(self, n, m):
        if n == 0:
            return -1
        if n == 1:
            return 0
        return (self.LastRemaining_Solution(n - 1, m) + m) % (n)

    # 解法2,不用递归就不会有那个错误，顺利通过
    # 注意不要忘记特殊情况：m==0 or n == 0:
    def LastRemaining_Solution2(self, n, m):
        if m == 0 or n == 0:
            return -1
        indexs = list(range(n))
        sp = 0
        while (len(indexs) > 1):
            sp = (sp + m - 1) % len(indexs)  # 每当删除一个数之后，起始点应该往前退一个
            indexs.remove(indexs[sp])

        return indexs[0]

    # 求1+2+3+...+n:由于不能使用循环语句，以及判断语句，所以使用短路原理
    # and 这个运算符，a and b，当a和b都大于0时，输出最大的那个，当ab中有一个是小于0时，输出0
    # 根据与运算的性质，当第一个数已经为0或者为False的时候，那么后面的也就不会再进行判断，这就是短路求值原理，多用于代码的优化
    def Sum_Solution(self, n):
        sum = n
        sum = sum and self.Sum_Solution(n - 1)
        sum += n
        return sum

    # 不用加减乘除做加法
    # 题目描述：写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。

    def Add(self, num1, num2):
        num1 = ctypes.c_int32(num1).value
        num2 = ctypes.c_int32(num2).value
        while num1 != 0:
            n1 = ctypes.c_int32(num1 & num2).value
            n1 = ctypes.c_int32(n1 << 1).value
            n2 = ctypes.c_int32(num1 ^ num2).value
            num1 = n1
            num2 = n2
        return num2

    def StrToInt(self, s):
        if str(s).strip() == '':
            return 0

        # 判断符号
        fh = 0
        if s[0] == '-':
            fh = 1
        elif s[0] == '+':
            fh = 2

        if fh >= 1:
            s = s[1:]
        # 判断是否有e
        index = str(s).find('e')
        sum1 = 0
        base = 1
        if index == -1:
            index = len(s)

        for i in range(index):
            if str(s[index - i - 1]).isdigit():
                sum1 += base * int(s[index - i - 1])
                base *= 10
            else:
                return 0

        sum2 = 0
        base = 1
        for i in range(index + 1, len(s)):
            if str(s[index - i]).isdigit():
                sum2 += base * int(s[index - i])
                base *= 10
            else:
                return 0
        sum = sum1 * pow(10, sum2)
        if fh == 1:
            sum = -sum
        return sum

    # 数组中重复的数字
    # 题目要求：这里要特别注~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    # 神操作：即链接：https://www.nowcoder.com/questionTerminal/623a5ac0ea5b4e5f95552655361ae0a8
    # 题目里写了数组里数字的范围保证在0 ~ n-1 之间，所以可以利用现有数组设置标志，当一个数字被访问过后，可以设置对应位上的数 + n，之后再遇到相同的数时，会发现对应位上的数已经大于等于n了，那么直接返回这个数即可。
    def duplicate(self, numbers, duplication):
        for num in numbers:
            if num >= len(numbers):
                num -= len(numbers)
            if numbers[num] >= len(numbers):
                duplication[0] = num
                return True
            else:
                numbers[num] += len(numbers)
        return False

    # 构建乘积数组
    # 剑指offer上这题不能用numpy，直接用list就行
    def multiply(self, A):
        if A == None or len(A) == 0:
            return np.array([])

        b = [1]
        for i in range(1, len(A)):
            b.append(b[i - 1] * A[i - 1])

        temp = 1
        for i in range(len(A) - 1):
            temp *= A[len(A) - i - 1]
            if len(A) - i - 2 >= 0:
                b[len(A) - i - 2] *= temp

        return np.array(b)

    # 正则表达式匹配:使用动态规划来做
    def match(self, s, pattern):
        m, n = len(s), len(pattern)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        for i in range(m + 1):  # 这个地方从0开始是为了防止s为空串的情况
            for j in range(1, n + 1):
                if pattern[j - 1] == '*':
                    dp[i][j] = dp[i][j - 2] or (
                            i > 0 and dp[i - 1][j] and (s[i - 1] == pattern[j - 2] or pattern[j - 2] == '.'))

                else:
                    dp[i][j] = i > 0 and (pattern[j - 1] == s[i - 1] or pattern[j - 1] == '.') and dp[i - 1][j - 1]
        return dp[m][n]

    def isNumeric(self, s):
        return re.match('^[+-]?[0-9]*(\.[0-9]+)?([eE][+-]?[0-9]+)?$', s) != None

    # 字符流中第一个不重复的字符
    # 由于一个字符的ascii不会超过256，因此，我们可以使用一个hash数组来存放输入字符当前的出现次数，其中ord()函数可以获取字符的ascii值。
    # 维护一个队列，记录单词的出现顺序，并且不断的更新，将出现次数大于1的早出现的字符都pop走
    def __init__(self):
        self.m = [0] * 256
        self.q = []

    def FirstAppearingOnce(self):
        while self.q and self.m[ord(self.q[0])] > 1:
            self.q.pop(0)
        if not self.q:
            return '#'

        return self.q[0]

    def Insert(self, char):
        self.m[ord(char)] += 1
        if self.m[ord(char)] == 1:
            self.q.append(char)

    # 链表中环的入口节点
    # 描述：给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。
    # 详情见pdf
    def EntryNodeOfLoop(self, pHead):
        p1, p2 = pHead, pHead
        flag = False
        while p1 and p2 and p1.next and p2.next.next:
            p1, p2 = p1.next, p2.next.next
            if p1 == p2:
                flag = True
                break
        if not flag: return None
        p1 = pHead
        while p1 and p2:
            if p1 == p2: break
            p1, p2 = p1.next, p2.next

        return p1

    class ListNode:

        def __init__(self, x):
            self.val = x
            self.next = None

    # 删除链表中的重复节点
    # 题目描述：在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5
    # 处理后为
    # 1->2->5
    def deleteDuplication(self, pHead):
        h1 = self.ListNode
        h1.val = 1
        h1.next = None
        p1, p2 = h1, pHead
        while p2:
            temp = p2.val
            temph = p2
            while p2 and p2.val == temp:
                p2 = p2.next
            if p2 == temph.next:  #
                p1.next = temph
                p1 = temph
        p1.next = None
        return h1.next

    # 解法2：使用递归
    def deleteDuplication2(self, pHead):
        if pHead == None:
            return None
        if pHead.next == None:
            return pHead

        current = pHead
        if current.next.val == pHead.val:
            while current and current.val == pHead.val:
                current = current.next
            self.deleteDuplication(current)

        else:
            pHead.next = self.deleteDuplication(pHead.next)
            return pHead

    # 二叉树的下一个节点
    # 分析：三种情况
    # 1.
    # 二叉树为空，则返回空；
    # 2.
    # 节点右孩子存在，则设置一个指针从该节点的右孩子出发，一直沿着指向左子结点的指针找到的叶子节点即
    # 为下一个节点；
    # 3.
    # 节点不是根节点。如果该节点是其父节点的左孩子，则返回父节点；否则继续向上遍历其父节点的父节点，
    # 重复之前的判断，返回结果。
    def GetNext(self, pNode):
        if pNode == None:
            return None

        if pNode.right == None:
            while pNode.next:
                p = pNode.next
                if p.left == pNode:
                    return p
                else:
                    pNode = pNode.next
            return None

        p = pNode.right
        while p.left != None:
            p = p.left
        return p

    # 对称的二叉树：请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

    def isSymmetrical(self, pRoot):
        if pRoot == None:
            return True

        leftTree = pRoot.left
        rightTree = pRoot.right
        queue1, queue2 = [], []
        queue1.append(leftTree)
        queue2.append(rightTree)
        while len(queue1) > 0 and len(queue2) > 0:
            if len(queue1) == len(queue2):
                length = len(queue1)
                node1 = queue1[0]
                node2 = queue2[0]
                queue1.remove(node1)
                queue2.remove(node2)
                length -= 1
                if (node1 == None and node2 != None) or (node1 != None and node2 == None):
                    return False
                if not (node1 or node2):
                    continue
                if node1.val == node2.val:
                    queue1.append(node1.left)
                    queue1.append(node1.right)
                    queue2.append(node2.right)
                    queue2.append(node2.left)
                else:
                    return False

            else:
                return False

        return True

    # 递归
    def isSymmetrical2(self, pRoot):
        if pRoot: return True
        return self.cMoot(pRoot.left, pRoot.right)

    def cMoot(self, tree1, tree2):
        if not tree1 or not tree2: return tree1 == tree2
        return tree1.val == tree2.val and self.cMoot(tree1.left, tree2.right) and self.cMoot(tree1.right, tree2.left)

    # 按之字形顺序打印二叉树:这个题目要求返回的是这种形式:
    # [[0], [2, 1], [3, 4, 5, 6], [10, 9, 8, 7]]
    # 就是每层都用一个list
    def Print(self, pRoot):
        if pRoot is None: return []
        path = []
        array1 = [pRoot]
        array2 = []
        fx = 1
        while array1:
            items = []
            while array1:
                node = array1.pop()
                if node is None:
                    continue
                items.append(node.val)
                if fx == 1:
                    array2.append(node.left)
                    array2.append(node.right)
                else:
                    array2.append(node.right)
                    array2.append(node.left)
            fx = -fx
            array1 = array2
            array2 = []
            if len(items) > 0:
                path.append(items)
        return path

    # 题目:把二叉树打印成多行
    # 题目描述:从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。
    # 返回二维列表[[1,2],[4,5]]
    def Print2(self, pRoot):
        if pRoot is None: return []
        queue = [pRoot]
        queue2 = []
        while queue:
            layer = []
            path = []
            while queue:
                node = queue[0]
                queue.remove(node)
                if node is None: continue
                layer.append(node.val)
                queue2.append(node.left)
                queue2.append(node.right)
            queue = queue2
            queue2 = []
            if len(layer) > 0: path.append(layer)
        return path

    # 另一种解法:使用递归:
    # 主要思想：先预先给每个层都保存一个[]在path中，然后按照顺序插入，比如第一层的第一个节点的孩子节点是在第二个节点的孩子节点前面插入
    def Print3(self, pRoot):
        path = []
        self.Printlayer(pRoot, 1, path)
        return path

    def Printlayer(self, root, depth, arraylist):
        if root == None: return
        if depth > len(arraylist): arraylist.append([])
        arraylist[depth - 1].append(root.val)
        self.Printlayer(root.left, depth + 1, arraylist)
        self.Printlayer(root.right, depth + 1, arraylist)

    # 序列化二叉树
    def Serialize(self, root):
        def doit(s, node):
            if node is None:
                s.append('#')
            else:
                s.append(str(node.val))
                doit(s, node.left)
                doit(s, node.right)

        s = []
        doit(s, root)
        return ' '.join(s)

    # 用的迭代器实现的:
    def Deserialize(self, s):
        def doit():
            val = next(vals)
            if val == '#': return None
            node = TreeLinkNode(int(val))
            node.left = doit()
            node.right = doit()
            return node

        vals = iter(s.split())
        return doit()

    # 二叉搜索树的第k个节点
    # 给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）    中，按结点数值大小顺序第三小结点的值为4。
    # 思想：通过中序列遍历可以获得排序的节点
    def KthNode(self, pRoot, k):
        def doit(node):
            if len(nodels) >= k: return nodels
            if node == None: return None
            doit(node.left)
            nodels.append(node)
            doit(node.right)
            return nodels

        if k == 0 or pRoot is None:
            return None
        nodels = []
        nodels = doit(pRoot)
        if k > len(nodels): return None
        return nodels[k - 1]

    # 解法2:借助中序遍历的过程：其中语句:if node:return 如果index已经达到了k，
    # 那么会被逐级返回，且通过if self.index == k返回第k个非None节点。如果还没到达k,则返回的都是None
    index = 0

    def KthNode2(self, pRoot, k):
        if pRoot is None: return None
        node = self.KthNode2(pRoot.left, k)
        if node: return node
        self.index += 1
        if self.index == k:
            return pRoot
        node = self.KthNode2(pRoot.right, k)
        if node: return node
        return None

    # 数据流中的中位数:总结:像这种数据流的一般都用队列等数据结构，使得能够直接获取到输出结果

    def __init__(self):
        self.minheap = []
        self.maxheap = []

    def Insert(self, num):
        if len(self.minheap) == 0 or num <= -self.minheap[0]:
            hp.heappush(self.minheap, -num)
        else:
            hp.heappush(self.maxheap, num)
        if len(self.minheap) == len(self.maxheap) + 2: hp.heappush(self.maxheap, -(hp.heappop(self.minheap)))
        if len(self.minheap) == len(self.maxheap) - 1: hp.heappush(self.minheap, -(hp.heappop(self.maxheap)))
        # print("="*20)
        # print(self.minheap)
        # print(self.maxheap)

    def GetMedian(self, n=None):  # 这个地方得添加一个n=None，才能通过，不然会提示：多给了一个参数
        if len(self.minheap) == len(self.maxheap):
            return (-self.minheap[0] + self.maxheap[0]) / 2.0  # 特别注意，这个地方一定要将数据类型转换为float类型，不然在牛客网上面会一直有数据类型的错误
        else:
            return -self.minheap[0]

    # 滑动窗口的最大值
    # 解法1:设置一个maxPoint指向当前最大值的index,如果过期了，就重新再寻找
    def maxInWindows(self, num, size):
        if num is None or size <= 0 or len(num) < size: return []  # 需要注意的是当数组长度小于size的时候，需要返回[]
        if len(num) == size: return [max(num)]  # 这个地方需要返回list
        maxPoint = num.index(max(num[:size]))
        arrayindex = [maxPoint]
        for i in range(size, len(num)):
            if i - maxPoint < size:
                if num[maxPoint] < num[i]: maxPoint = i
            else:
                maxPoint = num.index(max(num[i - size + 1:i + 1]))
            arrayindex.append(maxPoint)
        return list(map(lambda x: num[x], arrayindex))

    # 解法2:设置一个队列，每次一个元素在入队列之前，都会检查队列的第一个元素是否已经超过了窗口，如果超过，则删除
    # 然后再将队列中所有小于当前元素的元素都删除掉，最后将当前元素放入队列，队列的第一个元素就是当前窗口的最大值
    # 运行的时候这个解法的时间和空间效率都没有上一个解法好。。。。。
    def maxInWindows2(self, num, size):
        arrayindex = None
        maxnum = []
        for i in range(len(num)):
            if arrayindex is not None and len(arrayindex) > 0:
                if i - arrayindex[0] >= size: arrayindex.remove(arrayindex[0])

            if arrayindex is not None and len(arrayindex) > 0:
                if num[i] > num[arrayindex[len(arrayindex) - 1]]:
                    j = 0
                    while arrayindex and j < len(arrayindex):
                        if num[i] > num[arrayindex[j]]:
                            arrayindex.remove(arrayindex[j])
                        else:
                            j += 1
            else:
                arrayindex = []
            arrayindex.append(i)
            if i >= size - 1: maxnum.append(num[arrayindex[0]])
        return maxnum

    # 矩阵中的路径
    # 解法：使用dfs来解决
    def hasPath(self, matrix, rows, cols, path):
        visit = [False] * (rows * cols)
        for i in range(rows):
            for j in range(cols):
                if self.dfs(matrix=matrix, rows=rows, cols=cols, i=i, j=j, path=path, k=0, visit=visit):
                    return True
        return False

    def dfs(self, matrix, rows, cols, i, j, path, k, visit):
        index = i * cols + j
        if i >= 0 and i < rows and j >= 0 and j < cols:
            if visit[index] == False and path[k] == matrix[index]:
                k += 1
                if k == len(path):
                    return True
                visit[index] = True
                if self.dfs(matrix, rows, cols, i + 1, j, path, k, visit) or \
                        self.dfs(matrix, rows, cols, i - 1, j, path, k, visit) or \
                        self.dfs(matrix, rows, cols, i, j + 1, path, k, visit) or \
                        self.dfs(matrix, rows, cols, i, j - 1, path, k, visit):
                    return True
                visit[index] = False
        return False

    # 机器人的运动范围
    c = 0

    def movingCount(self, threshold, rows, cols):
        visit = [False] * (rows * cols)
        self.travel(threshold, rows, cols, 0, 0, visit)
        return self.c

    def travel(self, threshold, rows, cols, i, j, visit):
        index = i * cols + j
        if i == rows or j == cols or i == -1 or j == -1 or visit[index] == True:
            return visit

        if self.sum(str(i) + str(j)) <= threshold:
            self.c += 1
        else:
            return visit  # 如果这个方向已经达到阈值，则这个方向就终止了
        visit[index] = True
        visit = self.travel(threshold, rows, cols, i, j - 1, visit)
        visit = self.travel(threshold, rows, cols, i, j + 1, visit)
        visit = self.travel(threshold, rows, cols, i + 1, j, visit)
        visit = self.travel(threshold, rows, cols, i - 1, j, visit)
        return visit

    def sum(self, i):
        str1 = str(i)
        sum = 0
        for s in str1:
            sum += int(s)
        return sum

    # 顺时针打印矩阵

    def printMatrix(self, matrix):
         if matrix:
            return list(matrix.pop(0)) + self.printMatrix(list(zip(*matrix))[::-1])
         return []

    #把数组排成最小的数
    def PrintMinNumber(self, numbers):
        for i in range(len(numbers)):
            for j in range(i,len(numbers)):
                if not self.compareTo(numbers[i],numbers[j]):
                   numbers[i],numbers[j] = numbers[j],numbers[i]
        return ''.join(map(lambda x:str(x),numbers))


    def compareTo(self,s1,s2):
        str1 = str(s1) + str(s2)
        str2 = str(s2) + str(s1)
        return str1 < str2










s = Solution()
root = TreeLinkNode(5)
left = TreeLinkNode(3)
right = TreeLinkNode(7)
root.left = left
root.right = right

left1 = TreeLinkNode(2)
right1 = TreeLinkNode(4)
left.left = left1
left.right = right1

left2 = TreeLinkNode(6)
right2 = TreeLinkNode(8)
right.left = left2
right.right = right2

# left3 = TreeLinkNode(7)
# right3 = TreeLinkNode(8)
# left1.left = left3
# left1.right = right3
#
# left4 = TreeLinkNode(9)
# right4 = TreeLinkNode(10)
# right1.left = left4
# right1.right = right4

# print(s.Serialize(root)


# s.Insert(5)
# print(s.GetMedian())
# s.Insert(2)
# print(s.GetMedian())
# s.Insert(3)
# print(s.GetMedian())
# s.Insert(4)
# print(s.GetMedian())
# s.Insert(1)
# print(s.GetMedian())
# s.Insert(6)
# print(s.GetMedian())

# matrix = ['a','b','c','e','s','f','c','s','a','d','e','e']
# path='bcced'
# print(s.hasPath(matrix=matrix,rows=3,cols=4,path=path))

# print(s.xzmatrix([[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]))