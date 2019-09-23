# 笛卡尔积
# from itertools import product
# for x,y,z in product(['a','b','c'],['d','e','f'],['m','n']):
#     print(x+y+z)

# 获取对象信息
class textOp(object):
    def __init__(self):
        self.__text = "I have a dream"
        self.output = "OK"

    def __len__(self):
        return len(self.output)

    def __add__(self, other):
        self.output += other.output
        return self.output


if __name__ == "__main__":
    tObject = textOp()
    print(len(tObject))
    if hasattr(tObject, 'output'):
        setattr(tObject, 'output', "Martin Luther King")
    print(len(tObject))
    fObject = textOp()
    tObject + fObject
    print(len(tObject))
