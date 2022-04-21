class Compare(str):
    def __lt__(x, y):
        print(x+y, y+x)
        return x + y < y + x

class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        # 任意两个数 按照x + y < y + x比较即可
        # 5>34, 34>30, 3>30, 3<34, 3>30, 9>34, 9>5得到最终结果
        nums2str = map(str, nums)
        largest_num = ''.join(sorted(nums2str, key=Compare, reverse=True))
        return '0' if largest_num[0] == '0' else largest_num
        
