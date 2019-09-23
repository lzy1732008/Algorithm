import numpy as np
from leetcode.Tree import TreeNode

# 55、Jump Game
# 解法1:使用动态规划:但是超时了

def canJump(self, nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    steps = [False for _ in range(len(nums))]
    steps[0] = True
    for i in range(1, len(nums)):
        for j in range(i):
            steps[i] = steps[i] or (steps[j] and nums[j] >= (i - j))
            if steps[i]: break
    return steps[len(nums) - 1]


# 解法2:先求出能走的最大距离:贪心解法
def canJump(self, nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    maxdis = 0
    for i in range(len(nums)):
        if maxdis >= i:  # 说明当前这个位置可达
            maxdis = max(nums[i] + i, maxdis)
    return maxdis >= len(nums) - 1


# 53. Maximum Subarray
def maxSubArray(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    maxsum = -100000000000000
    point = 0
    for n in nums:
        point += n
        maxsum = max(maxsum, point)
        if point <= 0:  # 说明前面的子串对后面没有帮助了
            point = 0
    return maxsum


# 使用动态规划的思想来做:这个超时。。不知道为什
def maxSubArray2(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    localmax = [0] * (len(nums) + 1)
    globalmax = [0] * (len(nums) + 1)
    localmax[0] = nums[0]
    globalmax[0] = nums[0]
    for i in range(1, len(nums)):
        localmax[i] = max(localmax[i - 1] + nums[i], nums[i])  # 当前位置能获得的最大值有两种情况：1、上一个继承，2、从上一个开始断了，所以就是nums[i]
        globalmax[i] = max(localmax[i], globalmax[i - 1])
    return globalmax[len(nums) - 1]


# 121. Best Time to Buy and Sell Stock
# 由于必定是低价买入，高价卖出，所以高价必定从后往前找
def maxProfit(self, prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    if prices is None or len(prices) == 1: return 0
    maxPrice = prices[-1]
    reward = 0
    for i in reversed(range(len(prices))):
        maxPrice = max(prices[i], maxPrice)
        reward = max(reward, maxPrice - prices[i])
    return reward


# 使用全局最优和局部最优来做，
# 局部最优指的是在i时刻卖出的局部最优
def maxProfit(self, prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    if prices is None or len(prices) == 1: return 0
    globalr, localr = 0, 0
    for i in range(1, len(prices)):
        localr = max(0, localr + prices[i] - prices[i - 1])  # 这个local其实是第i时刻局部最优
        globalr = max(globalr, localr)
    return globalr


# 64. Minimum Path Sum
def minPathSum(grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    if grid is None or len(grid) == 0: return 0
    dp = []
    for i in range(len(grid)):
        temp = []
        for j in range(len(grid[0])):
            temp.append(float('inf'))
        dp.append(temp)

    dp[0][0] = grid[0][0]
    for i in range(0, len(grid)):
        for j in range(0, len(grid[0])):
            if i + j == 0: continue
            if i - 1 >= 0: dp[i][j] = min(dp[i][j], dp[i - 1][j])
            if j - 1 >= 0: dp[i][j] = min(dp[i][j], dp[i][j - 1])
            dp[i][j] += grid[i][j]
    return dp[-1][-1]


# 120. Triangle
def minimumTotal(triangle):
    """
    :type triangle: List[List[int]]
    :rtype: int
    """
    dp = []
    for i in range(len(triangle)):
        temp = []
        for j in range(len(triangle[i])):
            temp.append(float('inf'))
        dp.append(temp)
    dp[0][0] = triangle[0][0]
    for i in range(1, len(triangle)):
        for j in range(len(triangle[i])):
            if i + j == 0: continue
            if j - 1 >= 0: dp[i][j] = min(dp[i][j], dp[i - 1][j - 1])
            if j < len(triangle[i - 1]): dp[i][j] = min(dp[i][j], dp[i - 1][j])
            dp[i][j] += triangle[i][j]
    return min(dp[-1])


class DP:
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        dp = []
        for i in range(m):
            line = []
            for j in range(n):
                line.append(0)
            dp.append(line)

        dp[0][0] = 1
        for i in range(m):
            for j in range(n):
                if i + j == 0: continue
                if i - 1 >= 0: dp[i][j] += dp[i - 1][j]
                if j - 1 >= 0: dp[i][j] += dp[i][j - 1]
        return dp[-1][-1]

    #     63. Unique Paths II
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        if obstacleGrid is None or len(obstacleGrid) == 0: return 0
        if obstacleGrid[0][0] == 1:
            return 0
        dp = []
        for i in range(len(obstacleGrid)):
            line = []
            for j in range(len(obstacleGrid[0])):
                line.append(0)
            dp.append(line)
        dp[0][0] = 1
        for i in range(len(obstacleGrid)):
            for j in range(len(obstacleGrid[0])):
                if i + j == 0 or obstacleGrid[i][j] == 1: continue
                if i - 1 >= 0: dp[i][j] += dp[i - 1][j]
                if j - 1 >= 0: dp[i][j] += dp[i][j - 1]
        return dp[-1][-1]

    #     309. Best Time to Buy and Sell Stock with Cooldown:可以使用图来表达买、卖、等待这三个图的状态，用s表示目前的积蓄
    # 这题的解法非常好，思想可以借鉴
    def maxProfit309(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        s0 = [0] * len(prices)
        s1 = [0] * len(prices)
        s2 = [0] * len(prices)
        s0[0] = 0
        s1[0] = -prices[0]
        s2[0] = -10000000000
        for i in range(1, len(prices)):
            s0[i] = max(s0[i - 1], s2[i - 1])
            s1[i] = max(s0[i - 1] - prices[i], s1[i - 1])
            s2[i] = s1[i - 1] + prices[i]
        return max([s0[-1], s1[-1], s2[-1]])

    # 122. Best Time to Buy and Sell Stock II
    # 这题和上一题不一样的时，没有冷却时间，所以少了一个状态
    def maxProfit122(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if prices is None or len(prices) == 0: return 0
        s0 = [0] * len(prices)
        s1 = [0] * len(prices)
        s0[0] = 0
        s1[0] = -prices[0]
        for i in range(1, len(prices)):
            s0[i] = max(s0[i - 1], s1[i - 1] + prices[i])
            s1[i] = max(s1[i - 1], s0[i - 1] - prices[i])
        return max(s0[-1], s1[-1])

    # 解法2
    def maxProfit122_2(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if prices is None or len(prices) == 0: return 0
        maxpro = 0
        for i in range(len(prices)):
            if i + 1 < len(prices):
                temp = prices[i + 1] - prices[i]
                if temp > 0: maxpro += temp
        return maxpro

    #123. Best Time to Buy and Sell Stock III
    def maxProfit123(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        buy1,buy2 = -10000000, -10000000
        sell1, sell2 = 0, 0
        for p in prices:
            sell1 = max(sell1, buy1 + p)
            buy1 = max(buy1,  - p)

            sell2 = max(sell2, buy2 + p)
            buy2 = max(buy2, sell1 - p)
        return sell2



    # 91. Decode Ways:这题隐含约束是'01'等这种不能表示
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s is None or len(s) == 0: return 0
        first = 1
        second = 0 if s[0] == '0' else 1
        for i in range(2, len(s) + 1):
            cur = 0
            temp = s[i - 2: i]
            if '09' < temp < '27': cur += first
            if s[i - 1] != '0': cur += second
            first, second = second, cur
        return second

    # 639. Decode Ways II:碰到这种的确直接搞比较好
    def numDecodings639(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s is None or len(s) == 0: return 0
        first = 1
        if s[0] == '0':
            second = 0
        elif s[0] == '*':
            second = 9
        else:
            second = 1
        one = {'1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '9': 1, '*': 9}
        two = {'10': 1, '11': 1, '12': 1, '13': 1, '14': 1, '15': 1, '16': 1, '17': 1, '18': 1, '19': 1, '20': 1,
               '21': 1,
               '22': 1, '23': 1, '24': 1, '25': 1, '26': 1, '*0': 2, '*1': 2, '*2': 2, '*3': 2, '*4': 2, '*5': 2,
               '*6': 2,
               '*7': 1, '*8': 1, '*9': 1, '1*': 9, '2*': 6, '**': 15}

        for i in range(2, len(s) + 1):
            cur = 0
            temp = s[i - 2: i]
            p1 = one.get(s[i-1])
            p2 = two.get(temp)
            if p1: cur += p1 * second
            if p2: cur += p2 * first
            first, second = second, cur
        return second % 1000000007


#     95. Unique Binary Search Trees II
    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
    def wordBreak(self, s, wordDict):
        path = ''
        allpath = []
        self.wordBreakHelp(s, wordDict, path, allpath)
        return allpath

    def wordBreakHelp(self, s, wordDict, path, allpath):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        if len(s) == 0:
            allpath.append(path)

        for word in wordDict:
            temp = path
            if word == s[:len(word)]:
                if temp == '':
                    temp += word
                else:
                    temp += ' '+ word
                allpath = self.wordBreakHelp(s[len(word):],wordDict,temp, allpath)

        return allpath

#337. House Robber III
    #这个写的是用数组来存放树
    def robNum(self, root):
        rob = [0 for _ in range(len(root))]
        unrob = [0 for _ in range(len(root))]
        def visit(index):
            leftChild = 2 * index + 1
            rightChild = 2 * index + 2
            if (leftChild >= len(root) or root[leftChild] == None)\
                    and (rightChild >= len(root) or root[rightChild] == None):
                rob[index] = root[index]
            if leftChild < len(root) and root[leftChild] != None and (rightChild >= len(root) or root[rightChild] == None):
                visit(leftChild)
                unrob[index] = max(rob[leftChild], unrob[leftChild])
                rob[index] = unrob[leftChild] + root[index]
            if rightChild < len(root) and root[rightChild] != None and (leftChild >= len(root) or root[leftChild] == None):
                visit(rightChild)
                unrob[index] = max(rob[rightChild], unrob[rightChild])
                rob[index] = unrob[rightChild] + root[index]
            if (leftChild < len(root) and root[leftChild] != None)\
                    and (rightChild < len(root) and root[rightChild] != None):
                visit(leftChild)
                visit(rightChild)
                unrob[index] = max(rob[leftChild], unrob[leftChild]) + max(rob[rightChild], unrob[rightChild])
                rob[index] = unrob[leftChild] + unrob[rightChild] + root[index]
            return
        visit(0)
        return max(rob[0],unrob[0])

    def rob(self, root):
        def visited(node):
            if node is None:
                return (0,0)
            if node.left is None and node.right is None:
                return (0,node.vale)
            if node.left is None:
                rightRes = visited(node.right)
                unRob = max(rightRes)
                rob = rightRes[0] + node.val
                return (unRob, rob)
            if node.right is None:
                leftRes = visited(node.left)
                unRob = max(leftRes)
                rob = leftRes[0] + node.val
                return (unRob, rob)

            leftRes = visited(node.left)
            rightRes = visited(node.right)
            unRob = max(leftRes) + max(rightRes)
            rob = leftRes[0] + rightRes[0] + node.val
            return (unRob,rob)
        res = visited(root)
        return max(res)

    def maxProdcut(self,nums):
        if len(nums) == 0:
            return 0
        pre_min = nums[0]
        pre_max = nums[0]
        globalMax = nums[0]
        for n in nums[1:]:
            temp = (pre_max * n, pre_min * n, n)
            min_num = min(temp)
            max_num = max(temp)
            pre_min = min_num
            pre_max = max_num
            globalMax = max(globalMax, pre_max, pre_min)
        return globalMax

#     221. Maximal Square
    def maximalSquare(self, matrix):
        row = len(matrix)
        col = len(matrix[0])
        nums = [[int(n) for n in col] for col in matrix]
        res = 0
        for i in range(row):
            for j in range(col):
                if matrix[i][j] == '1':
                    if i - 1 >= 0 and j - 1 >= 0 and matrix[i - 1][j] == '1' and matrix[i][j - 1] == '1' and matrix[i - 1][j - 1] == '1':
                        nums[i][j] += min(nums[i - 1][j], nums[i][j - 1], nums[i - 1][j - 1]) #因为要考虑到[["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]这个例子里面交叉的情况
                    res = max(res, nums[i][j])
        return res * res

    #338. Counting Bits
    def countBits(self, num):
        nums = [0 for _ in range(num+1)]
        nums[1] = 1
        id = 0
        for i in range(2,num+1):
            if self.isPowerOfTwo(i):
                id = 0
            nums[i] = 1 + nums[id]
            id += 1
        return nums

    def isPowerOfTwo(self,n):
        return n & (n - 1) == 0

    #279. Perfect Squares
    def numSquares(self, n): #这个解决方案超时
        nums = [0]
        for i in range(1,n + 1):
            t = 1
            minTemp = i
            while t * t <= i:
                minTemp = min(minTemp, 1 + nums[i - t * t])
                t += 1
            nums.append(minTemp)
        return nums[-1]

    def numSquares2(self, n):
        squares = []
        base = 1
        while base**2 < 2**32 -1 and base**2 < n:
            squares.append(base**2)
            base += 1
        if squares[-1] == n:return 1
        squares = sorted(squares,reverse=True)
        self.result = n
        self.dfs(squares, 0, 0, n)
        return self.result

    def dfs(self, bases, currentSum, currentCnt, n):
        if currentSum == n:
            self.result = currentCnt
        for i, v in enumerate(bases):
            if currentSum + v <= n and currentSum + v * (self.result - currentCnt) > n:
                self.dfs(bases[i:], currentSum + v, currentCnt + 1, n)

#39. Combination Sum
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        self.path = []
        candidates = sorted(candidates,reverse=True)
        def dfs(bases, currentSum, currentPath):
            if currentSum == target:
                self.path.append(currentPath)
                return
            for i, v in enumerate(bases):
                if currentSum + v <= target:
                    dfs(bases[i:], currentSum + v, currentPath+[v])
        dfs(candidates,0,[])
        return self.path

    # 312. Burst Balloons
    def maxCoins(self, nums):
        n = len(nums)
        dp = [[0 for _ in range(len(nums) + 2)] for _ in range(len(nums) + 2)]
        nums = [1] + nums +[1]
        for l in range(1, n + 1):
            for left in range(1, n - l + 2):
                right = left + l - 1
                for k in range(left,right+1,1):
                    dp[left][right] = max(dp[left][right], dp[left][k-1] + dp[k+1][right] + nums[left-1] * nums[k] * nums[right + 1])
        return dp[1][n]

    #310. Minimum Height Trees
    #题解：https://www.cnblogs.com/grandyang/p/5000291.html
    def findMinHeightTrees(self, n, edges):
        graph = [[] for _ in range(n)]
        for e in edges:
            graph[e[0]].append(e[1])
            graph[e[1]].append(e[0])

        count = 0
        queue = []
        trash = []
        while n - count >2:
            queue = []
            for i, g in enumerate(graph):
                if len(g) == 1 and i not in trash:
                    queue.append(i)
            count += len(queue)

            for q in queue:
                for i, v in enumerate(graph):
                    if q in graph[i]:
                        graph[i].remove(q)
            trash.extend(queue)

        res = set([i for i in range(n)])
        res = res - set(trash)
        return res
#410. Split Array Largest Sum
    def splitArray(self,nums, m):
        def isValid(mid):
            cut, cutSum = 0, 0
            for i in range(len(nums)):
                cutSum += nums[i]
                if cutSum > mid:
                    cut, cutSum = cut + 1, nums[i]
            cut += 1
            return cut <= m

        low, high, ans = max(nums), sum(nums), -1
        while low <= high:
            mid = (low + high) // 2
            if isValid(mid):
                ans, high = mid, mid - 1
            else:
                low = mid + 1
        return ans






import sys
if __name__ == '__main__':
    dp = DP()
    nums = [7, 2, 5, 10, 8]
    m = 2
    print(dp.splitArray(nums, m))

    # dp = DP()
    # # candidates = [3,1,5,8]
    # # n = 8
    # # edges = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [4, 6], [6, 7]]
    # # res = dp.findMinHeigdp = DP()htTrees(n, edges)
    # # print(res)
    # input = [[1,2,4],['-','-',4],[7,6,5]]
    # path = dp.fun(input)
    # print(path)
