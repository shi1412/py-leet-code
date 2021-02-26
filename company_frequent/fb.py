from collections import deque
from collections import Counter

class palindrome_permutation_266:
    def canPermutePalindrome(self, s):
        count = 0
        a = Counter(s)
        for key, value in a.items():
             count += value % 2
        
        return count <= 1 

class moving_average_from_data_stream_346:
    def __init__(self, size):
        self.size = size
        self.queue = deque()
        self.sum = 0
        self.count = 0
        
    def next(self, val):
        self.count += 1
        self.queue.append(val)
        if self.count > self.size:
            tail = self.queue.popleft()
        else:
            tail = 0
        
        self.sum = self.sum - tail + val
        return self.sum / min(self.size, self.count)

class island_perimeter_463:
    def islandPerimeter(self, grid):
        row = len(grid)
        col = len(grid[0])
        res = 0
        
        for i in range(row):
            for j in range(col):
                if grid[i][j] == 1:
                    if i == 0:
                        up = 0
                    else:
                        up = grid[i - 1][j]
                    
                    if j == 0:
                        left = 0
                    else:
                        left = grid[i][j - 1]
                        
                    if i == row - 1:
                        down = 0
                    else:
                        down = grid[i + 1][j]
                        
                    if j == col - 1:
                        right = 0
                    else:
                        right = grid[i][j + 1]
                        
                    res += 4 - (up + down + left + right)
        
        return res            

class add_string_415:
    def addStrings(self, num1, num2):
        if num1 is None or num2 is None or num1 == "" or num2 == "":
            return num1 or num2
        
        res = []
        carry = 0
        p1 = len(num1) - 1
        p2 = len(num2) - 1
        while p1 >= 0 or p2 >= 0:
            if p1 >= 0:
                x1 = ord(num1[p1]) - ord('0')
            else:
                x1 = 0
            
            if p2 >= 0:
                x2 = ord(num2[p2]) - ord('0')
            else:
                x2 = 0
                
            value = (x1 + x2 + carry) % 10
            carry = (x1 + x2 + carry) // 10
            res.append(value)
            p1 -= 1
            p2 -= 1
        
        if carry:
            res.append(carry)
            
        return "".join(str(x) for x in res[::-1])
    
class average_of_levels_in_binary_tree_637:
    # DFS
    def averageOfLevels_A(self, root):
        if root is None:
            return []
        
        res = []
        count = []
        self.helper(root, 0, res, count)
        for i in range(len(res)):
            res[i] = res[i] / count[i]
            
        return res
    
    def helper(self, root, i, res, count):
        if root is None:
            return
        if i < len(res):
            res[i] = res[i] + root.val
            count[i] = count[i] + 1
        else:
            res.append(1.0 * root.val)
            count.append(1)
            
        self.helper(root.left, i + 1, res, count)
        self.helper(root.right, i + 1, res, count)

class merge_two_binary_trees_617:
    def mergeTrees(self, root1, root2):
        if root1 is None:
            return root2
        
        if root2 is None:
            return root1
        
        root1.val += root2.val
        root1.left = self.mergeTrees(root1.left, root2.left)
        root2.right = self.mergeTrees(root2.right, root2.right)
        return root1
        
class toeplitz_matrix_766:
    def isToeplitzMatrix(self, matrix):
        if matrix is None or len(matrix) == 0 or len(matrix[0]) == 0:
            return False
        
        groups = {}
        row = len(matrix)
        col = len(matrix[0])
        for i in range(row):
            for j in range(col):
                if i - j not in groups:
                    groups[i - j] = matrix[i][j]
                elif groups[i - j] != matrix[i][j]:
                    return False
                
        return True
    
class goat_latin_824:
    def toGoatLatin(self, S):
        vowel = "aeiouAEIOU"
        str_list = S.split(" ")
        for i in range(len(str_list)):
            if str_list[i][0] in vowel:
                str_list[i] += "ma"
            else:
                str_list[i] = str_list[i][1:] + str_list[i][0] + "ma"
            
            str_list[i] += "a"*(i + 1)
        
        return " ".join(str_list)
        
class squares_of_a_sorted_array_977:
    def sortedSquares(self, nums):
        if nums is None or len(nums) == 0:
            return nums
        
        res = [0] * len(nums)
        left = 0
        right = len(nums) - 1
        for i in range(len(nums) - 1, -1, -1):
            if abs(nums[left]) < abs(nums[right]):
                s = nums[right]
                right -= 1
            else:
                s = nums[left]
                left += 1
            
            res[i] = s * s
        
        return res
        
class remove_all_adjacent_duplicates_in_string_1047:
    def removeDuplicates(self, S):
        if S is None or len(S) == 0:
            return ""
            
        queue = []
        for i in range(len(S)):
            if len(queue) != 0 and queue[-1] == S[i]:
                queue.pop()
            else:
                queue.append(S[i])
        
        return "".join(queue)

class intersection_of_three_tree_sorted_arrays_1213:
    def arrayIntersection(self, arr1, arr2, arr3):
        res = []
        p1 = p2 = p3 = 0
        while p1 < len(arr1) and p2 < len(arr2) and p3 < len(arr3):
            if arr1[p1] == arr2[p2] == arr3[p3]:
                res.append(arr1[p1])
                p1 += 1
                p2 += 1
                p3 += 1
            else:
                if arr1[p1] < arr2[p2]:
                    p1 += 1
                elif arr2[p2] < arr3[p3]:
                    p2 += 1
                else:
                    p3 += 1
                    
        return res   
    
class maximum_nesting_depth_of_the_parentheses_1614:
    def maxDepth(self, s):
        if s is None or len(s) == 0:
            return 0
        
        res = cur = 0
        for i in s:
            if i == "(":
                cur += 1
                res = max(res, cur)
            elif i == ")":
                cur -= 1
                
        return res