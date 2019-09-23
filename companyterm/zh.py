def maxK(m,nums):
    if len(nums) == 1:
        return [nums[0]+m,nums[0]+m]

    temp = sorted(nums)
    t = m
    while t > 0:
        temp[0] += 1
        i = 1
        p = temp[0]
        while i < len(nums):
            if temp[i] < p:
                temp[i-1] = temp[i]
            else:
                temp[i-1] = p
                break
            i += 1
        t -= 1
    minK = temp[-1]

    temp = sorted(nums)
    maxK = temp[-1] + m
    return [minK, maxK]



def MaxCh(n):
    sum = 0
    nums = [0 for _ in range(n)]
    nums[0] = 1
    nums[1] = 1
    for i in range(6,n+1):
        sum += ChoHelper(i-6,nums)
    return sum

def ChoHelper(m,nums):
    if nums[m] == 0:
        sum = 0
        for i in range(1, m + 1):
            sum += ChoHelper(m - i,nums)
        nums[m] = sum
    return nums[m]



import sys
if __name__ == "__main__":
    n = int(sys.stdin.readline().strip())
    m = int(sys.stdin.readline().strip())
    i = 0
    nums = []
    while i < n:
        t = int(sys.stdin.readline().strip())
        nums.append(t)
        i += 1
    re = maxK(m,nums)
    print(str(re[0])+' '+str(re[1]))
    # print(MaxCh(n))

