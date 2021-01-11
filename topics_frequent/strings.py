class Solution:
    def minWindow(self, s: str, t: str) -> str:
        count = [0]*256
        for char in t:
            count[ord(char)] += 1
 
        n = len(t)
        j = 0
        res = float('inf')
        start = 0
        for i in range(len(s)):
            if count[ord(s[i])] > 0:
                n -= 1
            
            count[ord(s[i])] -= 1
                
            while n == 0:
                if i - j + 1 < res:
                    res = i - j +1
                    start = j
                    
                count[ord(s[j])] += 1
                if count[ord(s[j])] > 0:
                    n += 1
                
                j += 1
                    
        return "" if res == float('inf') else s[start: start+res]
    
test = Solution()
print(test.minWindow('ADOBECODEBANC', 'ABC'))