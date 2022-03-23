class Solution(object):
    def findMinArrowShots(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """

        if points is None:
            return 0

        points.sort(key = lambda x: x[1]) # 已经排序过，就可以用贪心来做!!!
        pos = points[0][1]
        ans = 1
        for ballon in points:
            if ballon[0] > pos:
                pos = ballon[1]
                ans += 1

        return ans
