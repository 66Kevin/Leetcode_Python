from collections import Counter

class Solution:
    def wordSubsets(self, A: List[str], B: List[str]) -> List[str]:

        # 通过最大频率构建新单词：['eko', 'eooo'] -> ['ekooo']，只需统计每个词中的最大频率字母拼接起来即得到新单词
        target = defaultdict(int)
        for word in B:
            for c, v in Counter(word).items():
                target[c] = max(target[c], v)
                # print(target)

        res = []
        for word in A:
            ct = Counter(word)
            for c, v in target.items():
                # 如果新单词中的某个单词在A中不存在 或者 新单词中的某个单词出现的频率高于A中某单词则跳过
                if c not in ct or v > ct[c]:
                    break
            else:
                res.append(word)
        return res
