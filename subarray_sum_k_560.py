class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        num_times = collections.defaultdict(int)
        num_times[0] = 1  # 先给定一个初始值，代表前缀和为0的出现了一次
        cur_sum = 0  # 记录到当前位置的前缀和
        res = 0
        for i in range(len(nums)):
            cur_sum += nums[i]  # 计算当前前缀和
            if cur_sum - k in num_times: 
                res += num_times[cur_sum - k]
            num_times[cur_sum] += 1
        return res

        # 我们维护一个Counter的哈希表dic，并初始pre_sum = 0
        # 循环数组，每次叠加pre_sum
        # 然后每次判断pre_sum - k在哈希表中值的数量，ret += dic[pre_sum - k]
        # 最终dic[pre_sum] += 1
        # 累加值total，就是最终的结果了。

