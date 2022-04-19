# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findMode(self, root: TreeNode) -> List[int]:

        if not root:
            return []

        self.dicts = collections.Counter()
        self.inOrder(root)

        return [k for k, v in self.dicts.items() if max(self.dicts.values())==v]

    def inOrder(self, root):
        if not root:
            return 
        self.inOrder(root.left)
        self.dicts[root.val] += 1
        self.inOrder(root.right)


