class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:

        path = []
        ans = []

        def backtracking(startIndex):
            if sum(path) == target:
                ans.append(path[:])
                return
            if sum(path) > target:
                return
            
            for i in range(startIndex, len(candidates)):
                path.append(candidates[i])
                backtracking(i)
                path.pop()

        backtracking(0)

        return ans 

