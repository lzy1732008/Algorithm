from jzoffer.ListNode import TreeLinkNode,RandomListNode
class Solution:
    def reConstructBinaryTree(self, pre, tin):
        root = self.reConstructBinaryTree2(pre, tin)
        return root

    def reConstructBinaryTree2(self, pre, tin):
        # write code here
        if pre is None or tin is None: return None
        root = TreeLinkNode(pre[0])
        midindex = tin.index(root.val)
        leftchs = tin[:midindex]
        rightchs = tin[midindex + 1:]
        if len(leftchs) == 0: root.left = None
        elif len(leftchs) == 1:
            node = TreeLinkNode(leftchs[0])
            root.left = node
        else:
            root.left = self.reConstructBinaryTree2(pre[1:len(leftchs)+1], leftchs)
        if len(rightchs) == 0: root.right = None
        elif len(rightchs) == 1:
            node = TreeLinkNode(rightchs[0])
            root.right = node
        else:
            root.right = self.reConstructBinaryTree2(pre[len(leftchs)+1:], rightchs)
        return root
    def HV(self, root):
        values = []
        node = []
        node.append(root)
        while node and len(node) > 0:
            temp = node[0]
            node = node[1:]
            values.append(temp.val)
            if temp.left: node.append(temp.left)
            if temp.right: node.append(temp.right)
        return values

    def __init__(self):
        self.Stack1 = []  # 存放
        self.Stack2 = []

    def push(self, node):
        # write code here
        self.Stack1.append(node)

    def pop(self):
        # return xx
        while self.Stack1:
            self.Stack2.append(self.Stack1.pop())
        self.Stack2.pop()
        while self.Stack2:
            self.Stack1.append(self.Stack2.pop())
        return self.Stack1

    def minNumberInRotateArray(self, rotateArray):
        # write code here
        if rotateArray is None or len(rotateArray) == 0: return 0
        for i in range(1, len(rotateArray)-1):
            if rotateArray[i-1] >= rotateArray[i]:
                return rotateArray[i]
        return rotateArray[-1]

    def NumberOf1(self, n):
        # write code here
        count = 0
        num = n
        while num != 0 and -(pow(2,31)-1)<=num<= (pow(2,32)-1):
            count += 1
            num = (num-1) & num
        if n < 0:
            return count + 1
        return count
#复杂链表的复制
    def Clone(self, pHead):
        if pHead is None: return None
        p = pHead
        #将每个复制结点插入到原结点后面
        while p:
            node = RandomListNode(p.label)
            node.next = p.next
            p.next = node
            p = node.next
        #将A'.ramdom = A.ramdom.next
        p = pHead
        while p:
            if p.random:
                p.next.random = p.random.next
            p = p.next.next
        #将链表分离
        p = pHead.next
        while p:
            if p.next:
                p.next = p.next.next
            p = p.next
        return pHead.next

    def __init__(self):
        self.point = None
        self.root = None

    def Convert(self, pRootOfTree):
        if pRootOfTree:
            if pRootOfTree.left is None and self.root is None:
                self.root = pRootOfTree
                self.point = pRootOfTree
                if pRootOfTree.right is None:
                    return
            if pRootOfTree.left:
                self.Convert(pRootOfTree.left)

            if self.root != pRootOfTree:
                self.point.right = pRootOfTree
                pRootOfTree.left = self.point
                self.point = pRootOfTree
            if pRootOfTree.right:self.Convert(pRootOfTree.right)

        return self.root

    # def Permutation(self, ss):
    #     # write code here
    #     pathls = []
    #     path = []
    #     self.visit(ss, pathls, path)
    #     return sorted(list(set(pathls)))
    #
    # def visit(self, ss, pathls, path):
    #     if len(ss) == 1:
    #         path.append(ss[0])
    #         pathls.append(''.join(path))
    #         return pathls
    #     for i in range(len(ss)):
    #         temp = [s for s in path]
    #         temp.append(ss[i])
    #         remain = ss[:i] + ss[i + 1:]
    #         self.visit(remain, pathls, temp)
    #     return pathls
#解法2
    def Permutation(self, ss):
        # write code here
        pathls = []
        path = []
        pathls = self.visit(list(ss), pathls, i=0)
        return sorted(list(set(pathls)))

    def visit(self, ss, pathls, i):
        if len(ss) == i + 1:
            pathls.append(''.join(ss))
        for j in range(i+1,len(ss)):
            ss[i],ss[j] = ss[j], ss[i]
            self.visit(ss,pathls,i+1)
            ss[i],ss[j] = ss[j], ss[i]
        return pathls

    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        # 使用快速排序进行解决
        if k <= 0 or tinput is None or len(tinput) == 0: return []
        if k > len(tinput): return []
        if k == len(tinput): return sorted(tinput)
        return list(sorted(self.QuickSort(tinput, low=0, high=len(tinput) - 1, k=k)))

    def Qsort(self, tinput, low, high):
        i = low
        key = tinput[low]
        while low < high:
            while low < high:
                if tinput[low] > key: break
                low += 1

            while low < high:
                if tinput[high] <= key: break
                high -= 1

            tinput[low], tinput[high] = tinput[high], tinput[low]
        tinput[i], tinput[low-1] = tinput[low-1],tinput[i]
        return low

    def QuickSort(self, tinput, low, high, k):
        if low < high:
            mid = self.Qsort(tinput, low, high)
            if mid < k - 1:
                self.QuickSort(tinput, low, mid-1, k)
                self.QuickSort(tinput, mid, high, k)
            elif mid > k - 1:
                self.QuickSort(tinput, low, mid-1, k)
            else:
                return tinput[:k]
        return tinput[:k]
#寻找和为tsum的连续数序列
    def FindContinuousSequence(self, tsum):
        # write code here
        numsls = []
        l = 1
        h = 2
        while l < h:
            thissum = (h + l) * (h - l + 1)/2
            if thissum > tsum: l += 1
            elif thissum < tsum: h += 1
            else:
                numsls.append([i for i in range(l,h+1)])
                l += 1
        return numsls
#和为S的两个数字
    def FindNumbersWithSum(self, array, tsum):
        # write code here
        re = []
        minM = 10000000000000
        l = 0
        h = len(array)-1
        while l < h:
            sum = array[l] + array[h]
            if sum == tsum:
                mul = array[l] * array[h]
                if mul < minM:
                    minM = mul
                    re = [array[l],array[h]]
                l += 1
            elif sum < tsum:
                l += 1
            else: h -= 1
        return re
    def isValidBST(self, root) -> bool:
        pre = -10000000000
        T = root
        stack = []
        while T or len(stack) > 0:
            while(T):
                stack.append(T)
                T = T.left

            if len(stack) > 0:
                T = stack.pop()
                if pre >= T.val:
                    return False
                pre = T.val
                T = T.right
        return True

    def isValidBST2(self, root) -> bool:
        def validate(node, lower, upper):
            if not node:
                return True
            if lower is not None and node.val <= lower: #注意：这里不能直接用low， 因为当low等于0的时候，会被默认等同于False
                return False
            if upper is not None and node.val >= upper:
                return False
            return validate(node.left, lower, node.val) and validate(node.right, node.val, upper)
        return validate(root, None, None)







s =Solution()
root = TreeLinkNode(0)
node1 = TreeLinkNode(2)
node2 = TreeLinkNode(-1)
# node3 = TreeLinkNode(4)
# node4 = TreeLinkNode(6)
root.right = node2

# root = s.Convert(root)
print(s.isValidBST2(root))

# arr = [1,2,4,7,11,15]
# pathls = s.FindNumbersWithSum(arr,15)
# print(pathls)















