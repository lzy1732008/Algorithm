
def fun1(nums):
    if nums is None or len(nums) < 3: return
    if len(nums) == 3: return nums[0] * nums[1] * nums[2]
    maxnum1 = -1000000
    maxnum2 = -1000000
    maxnum3 = -1000000
    minnum1 = 1000000
    minnum2 = 1000000

    for n in nums:
           if n > maxnum1:
               maxnum3 = maxnum2
               maxnum2 = maxnum1
               maxnum1 = n
               continue
           elif n > maxnum2:
               maxnum3 = maxnum2
               maxnum2 = n
               continue
           elif n > maxnum3:
               maxnum3 = n
               continue

    for n in nums:
        if n < minnum1:
            minnum2 = minnum1
            minnum1 = n
            continue
        elif n < minnum2:
            minnum2 = n
            continue


    print(maxnum1,maxnum2,maxnum3,minnum2,minnum1)

    if len(nums) == 4:
        if maxnum3 < 0:
            return maxnum1 * maxnum3 * minnum1
        else:
            return maxnum1 * maxnum2 * maxnum3
        if maxnum2 < 0:
            return maxnum1 * maxnum2 * maxnum3

    re1 = maxnum1 * maxnum2 * maxnum3
    re2 = maxnum1 * minnum1 * minnum2
    re = max(re1,re2)
    return re

#
def fun2(len,num1,num2):
    num1 = sorted(num1)
    num2 = sorted(num2,reverse=True)
    sum = 0
    for n1,n2 in zip(num1,num2):
        sum += n1 * n2
    return sum




def fun3(s):
    c_count={'a':0,'b':0,'c':0,'d':0,'e':0,'f':0,'g':0,'h':0,'i':0,'j':0,'k':0,'l':0,
             'm':0,'n':0,'o':0,'p':0,'q':0,'r':0,'s':0,'t':0,'u':0,'v':0,'w':0,'x':0,'y':0,'z':0}
    news = []
    for c in s:
        news.append(str(c).lower())

    for c in news:
        c_count[c]+= 1

    point = 0
    while point < len(s) -1:
        c = news[point]
        if c > news[point+1] and c_count[c] > 1:
            point += 1
        else:
            return news[point]

    return news[point]

def fun4(n,d,cites,values):
    p1 = 0
    p2 = n-1
    maxv = -1
    while cites[p2]-cites[p1] >= d:
        temp = values[p1] + values[p2]
        if temp > maxv: maxv = temp
        if values[p1] < values[p2]: p1 += 1
        else: p2 -= 1

    return maxv


def fun5(n,d,cites,values):
    start = 0
    point = 1
    #找到第一个可以抢的位置，point
    for i in range(n):
        if cites[i] - cites[0] >= d:
            point = i
            break
    #找到之后的最大价值
    maxv = -1
    while start < n:
        maxnum = max(values[point:n])
        #
        # for j in range(point, n):
        #     if values[j] > maxnum: maxnum = values[j]
        re = values[start] + maxnum
        if maxv < re: maxv = re
        start += 1

        while point < n:
            if cites[point]-cites[start] >= d:
                break
            point += 1
    return maxv




import sys
if __name__ == "__main__":
    line = sys.stdin.readline().strip().split()
    n = int(line[0])
    d = int(line[1])
    cites = []
    values = []
    for i in range(n):
        line2 = list(map(int,sys.stdin.readline().strip().split()))
        cites.append(line2[0])
        values.append(line2[1])
    maxv = fun5(n,d,cites,values)
    print(maxv)









