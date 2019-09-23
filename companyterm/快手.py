import companyterm.meituan as mt
def maxApproach():
    p2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    dp = [0 for _ in range(1001)]
    dp[0] = 1
    for i in range(1,1001):
        for j in range(10):
            if i - p2[j] < 0: break
            dp[i] += dp[i - p2[j]]
            dp[i] = dp[i] % 1000000003
    return dp


def NormalStr(s):
    chars = list(set(s))
    charDict = {}
    for c in chars:
        charDict[c] = str(s).count(c)
    t = charDict.items()
    s = list(sorted(t,key=lambda x:x[1]))
    alls = ''
    for e in s:
        alls += e[0]+str(e[1])
    return alls

def Compare(a,b):
    return sum(a)<sum(b)

def WorstFriend(m,arrays):
    allsubs = []
    start = 0
    end = 0
    localSum = 0
    while end < len(arrays):
        temp = arrays[end] + localSum
        if temp > localSum:
            localSum = temp
            end += 1
        else:
            allsubs.append([start,end])
            end += 1
            start = end
            localSum = 0
    if temp > 0:
        allsubs.append([start,end])

    #首先判断m是不是比所有正数还多
    allsum = 0
    for p in allsubs:
        allsum += p[1] - p[0]
    if allsum < m:
        #寻找所有负数
        negs = []
        for c in arrays:
            if c < 0:
               negs.append(c)
        negs = sorted(negs)
        s = 0
        for a in allsubs:
            s += sum(arrays[a[0]:a[1]])
        s += sum(negs[allsum - m - 1:])
        return s
    else:
        if m >= len(allsubs):
            s = 0
            for a in allsubs:
                s += sum(arrays[a[0]:a[1]])
            return s
        else:
            for i in range(len(allsubs)):
                allsubs[i] = arrays[allsubs[i][0]:allsubs[i][1]]
            allsubs = sorted(allsubs, key=lambda x, y: Compare(x, y))
            s = 0
            for i in range(m):
                s += s(allsubs[i])
            return s

def MaxSubStr(s1,s2):
    maxLen = 0
    for i in range(len(s1)):
        j = 0
        while j < len(s2):
            if s2[j] == s1[i]:
                end = i + 1
                t = j + 1
                while t < len(s2) and end < len(s1) and s2[t] == s1[end]:
                    t += 1
                    end += 1
                maxLen = max(maxLen, end-i)
            j += 1
    return maxLen


def LossNum(arrays):
    if len(arrays) == 0: return 0
    arrays = set(arrays)
    for i in range(len(arrays)+1):
        if i not in arrays:
            break
    return i

#最小生成树
def MinCostHelper(n,costs):
    costs = sorted(costs,key=lambda x:x[2])
    edges = []
    knows = [False for _ in range(n)]
    for c in costs:
        n1,n2,cost = c[0],c[1],c[2]
        flag = 0
        if knows[n1] == False:
            knows[n1] = True
            flag = 1

        if knows[n2] == False:
            knows[n2] = True
            flag = 1
        if flag == 1:
            edges.append(cost)
    return edges


def MinCost(n,m,vals,costs):
    edges = MinCostHelper(n,costs)
    minval = min(vals)
    mincost = sum(edges) + minval
    return mincost



import sys
if __name__ == "__main__":
    mt.happyFunction('pppep')
    # samples =int(sys.stdin.readline().strip())
    # for _ in range(samples):
    #     nm = list(map(int,sys.stdin.readline().strip().split(' ')))
    #     vals = list(map(int,sys.stdin.readline().strip().split(' ')))
    #     costs = []
    #     for _ in range(nm[0]):
    #         c = list(map(int,sys.stdin.readline().strip().split(' ')))
    #         for _ in range(2):
    #             c[_] = c[_] -  1
    #         costs.append(c)
    #
    #     mindis = MinCost(nm[0],nm[1],vals,costs)
    #     print(mindis)
    # nums  = list(map(int,sys.stdin.readline().strip().split(',')))



