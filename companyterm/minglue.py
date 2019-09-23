def format(input, n):
    output = []
    while input > 0:
        output.append(input % n)
        input = int(input / 2)
    return ''.join(map(str,list(reversed(output))))

import sys
if __name__ == "__main__":
    nums = list(map(int,sys.stdin.readline().strip().split()))
    res = format(nums[0],nums[1])
    print(res)