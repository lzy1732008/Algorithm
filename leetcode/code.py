class Solution:
    #1.two sum
    def twoSum(self, nums, target):
        keyindex = {}
        for i in range(len(nums)):
            if keyindex.get(nums[i]) is None:
                keyindex[nums[i]] = [i]
            else:
                keyindex[nums[i]].append(i)

        keyindex = sorted(keyindex.items(),key=lambda x:x[0]) #这时keyindex变成了一个list，每个元素是一个tuple，tuple的第一个元素是key,第二个元素是value
        indexs,nums = [],[]
        for e in keyindex:
            indexs += e[1]
            nums += [e[0]] * len(e[1])
        point1, point2 = 0, len(nums)-1
        while point1 < point2:
            sum = nums[point1] + nums[point2]
            if sum == target:
                return [indexs[point1],indexs[point2]]
            elif sum < target:
                point1 += 1
            else:
                point2 -= 1
        return []





