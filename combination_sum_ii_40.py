# class Solution:
#     def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:

#         path = []
#         ans = []

#         def backtracking(startIndex):
#             if sum(path) == target:
#                 ans.append(path[:])
#                 return 
#             if sum(path) > target:
#                 return
            
#             for i in range(startIndex, len(candidates)):
#                 path.append(candidates[i])
#                 backtracking(i+1)
#                 path.pop()
            
#             backtracking(0)

#             return ans
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        path = []
        def backtrack(startIndex):
            if sum(path) == target: 
                res.append(path[:])
            if sum(path) > target:
                return

            for i in range(startIndex,len(candidates)):  #要对同一树层使用过的元素进行跳过
                if i > startIndex and candidates[i] == candidates[i-1]: continue  #用startIndex来去重,要对同一树层使用过的元素进行跳过
                path.append(candidates[i])
                backtrack(i+1)  #i+1:每个数字在每个组合中只能使用一次
                path.pop()

        candidates = sorted(candidates)  #排序，让其相同的元素都挨在一起。
        backtrack(0)
        return res

