

def Mountain(highs):
    maxlen = 0
    head = 0
    mid = -1
    tail = 0
    for i in range(1,len(highs)):
        if i + 1 < len(highs):
            if highs[i+1] < highs[i] > highs[i-1]: mid = i
            if highs[i-1] > highs[i] :tail = i
        else:
            if highs[i-1]>highs[i]:tail = i
        if head < mid < tail: maxlen = max(maxlen, tail - head + 1)

        if i + 1< len(highs):
            if highs[i-1]> highs[i] < highs[i+1]:
               head = i
               mid = i
               tail = i
    if maxlen >= 3:
        return maxlen
    return 0





def MinK(m,n):
    if m == 0: return 0
    strls = list(map(str, range(1,m+1)))
    str_s = {"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[],"9":[]}
    for s in strls:
        str_s[s[0]].append(s)
    sum = 0
    for i in range(1,10):
        temp = sum + len(str_s[str(i)])
        if temp >= n:
            n_sorted = sorted(str_s[str(i)])
            return n_sorted[n-sum-1]
        else:
            sum = temp
    return 0

def fun5(n):
    all_seq = []
    nums = [(i+1) for i in range(n)]

    while(1):
        all_seq.append(nums)

        j = n - 2
        while nums[j] > nums[j + 1] and j >= 0:
            j -= 1

        if j < 0:
            break
        k = n - 1
        while k > j and nums[k] < nums[j]:
            k -= 1

        t = nums[k]
        nums[k] = nums[j]
        nums[j] = t

        l = j + 1
        r = n - 1

        while l < r:
            t = nums[l]
            nums[l] = nums[r]
            nums[r] = t
            l += 1
            r -= 1
    print(all_seq)




# input:
# 53941 38641 31525 75864 29026 12199 83522 58200 64784 80987
# 12199 29026 31525 38641 53941 58200 64784 75864 80987 83522
# class NumLink:
#     def __init__(self):
#         self.index = 0
#         self.value = 0
#
# def fun3(nums):
#     all_os, all_js = [],[] #偶数、奇数
#     for n in nums:
#         if n % 2 == 0:
#             all_os.append(n)
#         else:
#             all_js.append(n)
#
#     nums = sorted(nums)
#
fun5(4)





import sys
if __name__ == "__main__":
    nums = list(map(int, list(sys.stdin.readline().strip().split())))
    if len(nums) == 2:
        num1 = int(nums[0])
        num2 = int(nums[1])
        temp = MinK(num1, num2)
        print(temp)





