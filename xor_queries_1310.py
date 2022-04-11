class Solution:
    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        # xor(l,r)=xor(1,r) ⊕ xor(1,l−1)
        sz = len(arr)
        pre_sum = [0 for _ in range(sz+1)] # 0 异或另一个数还是 那个数本身
        for i in range(sz):
            pre_sum[i+1] = pre_sum[i] ^ arr[i]
        
        ans = []
        for left, right in queries:
            ans.append(pre_sum[right+1] ^ pre_sum[left]) # 异或的相对运算还是异或
        return ans
