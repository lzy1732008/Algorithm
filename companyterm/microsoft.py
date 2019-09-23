# Read only region start
class UserMainCode(object):
    @classmethod
    def maxCircles(cls, input1, input2, input3):
        '''
        input1 : int
        input2 : int
        input3 : int

        Expected return type : int
        '''
        # Read only region end
        # Write code here
        count = 0
        last = input2
        depth = 0
        return cls.maxHelper(cls,input1, input2, input3, count, last, depth)

    def maxHelper(cls, input1, input2, input3, count, last, depth):
        if depth >= input3:
            return count
        for i in range(1, input1 + 1):
            if i != last and (int(i / last) == i / last or int(last / i) == last / i):
                temp = depth + 1
                if i == input2:
                    count += 1
                count = cls.maxHelper(cls,input1, input2, input3, count, i, temp)
        return count

    @classmethod
    def fun(cls,input1,input2,input3):
        sum = 0
        input1 = [i+1 for i in range(input1+1)]
        for q in input3:
            if q[0] == 1:
                input1.remove(input1[0])
            if q[0] == 2:
                input1.remove(input1[q[1] - 1])
            if q[0] == 3:
                sum += input1.index(q[1]) + 1
        return sum

    def fun2(cls,input1,input2):
        count = 0
        return cls.helper(input1,input2,count)

    def helper(cls,input1,input2, count):
        for i in range(len(input2)):
            j = len(input2) - 1
            while j > i:
                if input2[i] == input2[j]:
                    # 查看两个最少需要的枪数
                    left = i + 1
                    temp = input2[left:j]
                    count += cls.helper(len(temp), temp, count)
                    break
                j -= 1
            count += i + len(input2) - j - 1
        return count


u = UserMainCode()
print(u.fun2(5,[1,4,3,1,5]))