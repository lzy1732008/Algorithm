import math

def ClearNum(n,k):
    #最大需要切分的个数
    count = 0
    num = n
    while num > 1:
        num = num / 2
        count += 1
    if count > k:
        return k + math.ceil(n/pow(2,k))
    else:
        return count + math.ceil(n/pow(2,k))

def NumFun(nums,k):
    nums = sorted(nums)
    #首先将0全部剔除
    index = 0
    lastnum = 0
    while k > 0 and index < len(nums):
        while index < len(nums):
            if nums[index] != lastnum:
                break
            index += 1
        print(nums[index] - lastnum)
        lastnum = nums[index]
        index += 1
        k -= 1


import sys
if __name__ == "__main__":
    parameters = list(map(int,sys.stdin.readline().strip().split()))
    allnums = list(map(int,sys.stdin.readline().strip().split()))
    NumFun(allnums,parameters[1])

