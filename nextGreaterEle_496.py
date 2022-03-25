class Solution(object):
    def nextGreaterElement(self, nums1, nums2):
        dic, stack = {}, []

        for i in range(len(nums2)):
            while stack and stack[-1] < nums2[i]:
                dic[stack.pop()] = nums2[i]
            stack.append(nums2[i])

        return [dic.get(x, -1) for x in nums1] # 如果字典中没有key，则默认返回-1

