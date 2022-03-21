import collections


def frequencySort(s):
    """
    :type s: str
    :rtype: str
    """
    # 构建字典统计频数
    count_dict = collections.Counter(s)
    # 创建桶数组
    bucket = [[] for _ in range(len(s) + 1)]
    # 按频数大小分别放进各个桶里,桶的索引(下标)为字符出现的频数,桶里的值为该字符
    for key, value in count_dict.items():
        bucket[value].append(key * value) # bucket: [[], ['t', 'r'], ['ee'], [], []]
    # 逆序读取桶里的字符，即按照频数大小降序排列读取
    res = []
    for index in range(len(bucket) - 1, -1, -1):
        if bucket[index]:
            res.extend(bucket[index])
    return ''.join(res)


if __name__ == '__main__':
    print(frequencySort("tree"))
