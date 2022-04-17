class Solution:
    def reverseWords(self, s: str) -> str:
        
        pt = len(s) - 1
        ans = []
        while pt >= 0:
            while s[pt] == ' ':
                pt -= 1
            snd = pt
            while s[snd] != ' ' and snd >=0:
                snd -= 1
            if s[snd+1: pt+1]:
                ans.append(s[snd+1: pt+1])
                ans.append(' ')
            pt = snd
        
        return ''.join(ans[0: len(ans)-1])

