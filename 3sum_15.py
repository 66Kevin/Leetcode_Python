class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:

        if not nums:
            return []

        ans = []
        path = []

        def backtracking(startIndex):
            if len(path) == 3:
                if sum(path) == 0 and path not in ans:
                    ans.append(path[:])
                return
            
            for i in range(startIndex, len(nums)):
                path.append(nums[i])
                backtracking(i+1)
                path.pop()

        nums.sort() # 一定先排序，保证生成的序列是按照字典顺序
        backtracking(0)
        return ans


