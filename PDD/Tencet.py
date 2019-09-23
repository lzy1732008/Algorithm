
def fun1(n,a,arry):
    mindis = 1000000
    step = 0
    return fun2(a,0,arry,step,n,mindis)

def fun2(last,sum,arry,step,n,mindis):
    if step == n-1 or arry is None:
        return sum
    for c in arry:
        newarray = []
        for p in arry:
            if p != c: newarray.append(p)

        re = fun2(c,sum+abs(c-last),newarray,step+1,n,mindis)
        if re < mindis:
            mindis = re
    return mindis


def fun3(h,towers):
    dp = []
    for i in range(h+1):
        dp[i] = 1000000
    for i in range(1,h):
        if i-2>=0: dp[i] = min(dp[i-2],dp[i])
        if i-1>=0: dp[i] = min(dp[i-1],dp[i])
        dp[i] = dp[i] + towers[i]
    return dp[h-1]

def fun4(n):
    ls = []
    nums = [i+1 for i in range(n)]
    while(len(nums)>=2):
        ls.append(nums[0])
        nums = nums[1:]
        temp = nums[0]
        nums.remove(temp)
        nums.append(temp)
    return ls+[nums[0]]



import sys
if __name__ == "__main__":
    line = int(sys.stdin.readline().strip())
    # lines = list(map(int, sys.stdin.readline().strip().split()))
    maxv = fun4(line)
    print(maxv)