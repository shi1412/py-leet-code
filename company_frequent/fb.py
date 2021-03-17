from collections import deque
from collections import Counter
import heapq

class isomorphic_strings_205:
    def isIsmorphic(self, s, t):
        if s is None or t is None:
            return True
         
        obj = {}
        for i in range(len(s)):
            a = s[i]
            b = t[i]
            if a in obj:
                if obj[a] == b:
                    continue
                else:
                    return False
            else:
                if b in obj.values():
                    return False
                else:
                    obj[a] = b
                    
        return True

class contains_duplicate_II_219:
    def containNearbyDuplicate(self, nums, k):
        if nums is None:
            return False
        
        res = set()
        for i in range(len(nums)):
            if nums[i] in res:
                return True
            
            res.add(nums[i])
            if len(res) > k:
                res.remove(nums[i - k])
        
        return False
        
class palindrome_permutation_266:
    def canPermutePalindrome(self, s):
        count = 0
        a = Counter(s)
        for key, value in a.items():
             count += value % 2
        
        return count <= 1 

class reverse_vowels_of_a_string_345:
    def reverseVowels(self, s):
        if s is None or len(s) == 0:
            return s
        
        vowles = "aeiouAEIOU"
        start = 0
        end = len(s) - 1
        res = list(s)
        while start < end:
            if res[start] in vowles and res[end] in vowles:
                res[start], res[end] = res[end], res[start]
                start += 1
                end -= 1
            elif res[start] in vowles:
                end -= 1
            else:
                start += 1
        
        return "".join(res)
        
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

class intersection_of_two_arrays_II_350:
    def intersect(self, nums1, nums2):
        nums1.sort()
        nums2.sort()
        i, j, k = 0, 0, 0
        while i < j:
            if nums1[i] < nums2[j]:
                i += 1
            elif nums1[i] > nums2[j]:
                j += 1
            else:
                nums1[k] = nums1[i]
                i += 1
                j += 1
                k += 1
        
        return nums1[0:k]

class ransom_note_383:
    def canConstruct(self, ransomNote, magazine):
        if len(magazine) < len(ransomNote):
            return False
        
        m_counts = Counter(magazine)
        r_counts = Counter(ransomNote)
        
        for key, value in r_counts.items():
            if not m_counts[key]:
                return False
            elif m_counts[key] and m_counts[key] < value:
                return False
            
        return True
              
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

class diameter_of_binary_tree_543:
    def diameterOfBinaryTree(self, root):
        self.res = 1
        self.helper(root)
        return self.res - 1
    
    def helper(self, root):
        if root is None:
            return 0
        
        L = self.helper(root.left)
        R = self.helper(root.right)
        self.res = max(self.res, L + R + 1)
        return max(L, R) + 1

class subtree_of_another_tree_572:
    def isSubTree(self, s, t):
        if not s and not t:
            return True
        elif not s or not t:
            return False
        
        return self.isSameTree(s, t) or \
            self.isSameTree(s.left, t) or \
            self.isSameTree(s.right, t)
                    
    def isSameTree(self, s, t):
        if not s and not t:
            return True
        elif not s or not t:
            return False
        
        return s.val == t.val and \
            self.isSameTree(s.right, t.right) and \
            self.isSameTree(s.left, t.left)
        
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

class valid_palindrome_II_680:
    def validPalindrome(self, s):
        if s is None or len(s) == 0:
            return False
        
        left, right = 0, len(s) -1
        while left < right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else: 
                return self.helper(s, left + 1 , right) or \
                    self.helper(s, left, right - 1)
        
        return True
        
    def helper(self, s, left, right):
        i, j = left, right
        while i < j:
            if s[i] != s[j]:
                return False
            
            i += 1
            j -= 1
            
        return True

class kth_largest_element_in_a_stream_703:
    def __init__(self, k, nums):
        self.heap = []
        self.k = k
        for num in nums:
            self.add(num)
            
    def add(self, val):
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
            
        return self.heap[0]
    
class binary_search_704:
    def search(self, nums, target):
        if nums is None or len(nums) == 0:
            return -1
        
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            
            if target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
                
        return -1 
        
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

class backspace_string_compare_844:
    def backspaceCompare(self, S, T):
        return self.helper(S) == self.helper(T)
    
    def helper(self, __str):
        stack = []
        for i in __str:
            if stack and i == "#":
                stack.pop()
            elif i != "#":
                stack.append(i)
                
        return "".join(stack)
    
class monotonic_array_896:
    def isMonotonic(self, A):
        increasing = decreasing = True
        for i in range(len(A) - 1):
            if A[i] > A[i + 1]:
                increasing = False
                
            if A[i] < A[i + 1]:
                decreasing = False
                
        return increasing or decreasing

class range_sum_of_bst_938:
    def rangeSumBST(self, root, low, high):
        self.res = 0
        self.l = low
        self.h = high
        self.helper(root)
        return self.res
    
    def helper(self, root):
        if root is None:
            return 0
        
        if root:
            if self.l <= root.val <= self.h:
                self.res += root.val
            
            if self.l < root.val:
                self.helper(root.left)
                
            if self.h > root.val:
                self.helper(root.right)
        
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
    
class add_to_array_form_of_integer_989:
    def addToArrayForm(self, A, K):
        n = len(A)
        res = []
        i, __sum, carry = n - 1, 0, 0
        
        while i >= 0 or K != 0:
            x = A[i] if i >= 0 else 0
            y = K % 10 if K != 0 else 0
            __sum = x + y + carry 
            carry = __sum // 10
            
            K = K // 10
            i -= 1
            res.append(__sum % 10)
            
        if carry != 0:
            res.append(carry)
        
        return res[::-1]
           
class cousins_in_binary_tree_993:
    def __init__(self):
        self.dep = None
        self.check = False
    
    def isCousins(self, root, x, y):
        self.helper(root, 0, x, y)
        return self.check

    def helper(self, node, dep, x, y):
        if node is None:
            return False
     
        if self.dep and dep > self.dep:
            return False
        
        if node.val == x or node.val == y:
            if self.dep is None:
                self.dep = dep
            
            return self.dep == dep
        
        left = self.helper(node.left, dep + 1, x, y)
        right = self.helper(node.right, dep + 1, x, y)
        if left and right and self.dep != dep + 1:
            self.check = True
            
        return left or right
    
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

class kth_missing_postive_number_1539:
    def findKthPositive(self, arr, k):
        if k <= arr[0] - 1:
            return k
        # if there is missing value at the beginning
        k -= arr[0] - 1
        for i in range(len(arr) - 1):
            # when the missing value is in the middle
            cur = arr[i + 1] - arr[i] - 1
            if k <= cur:
                return arr[i] + k
            
            # if k is at the end
            k -= cur
            
        return arr[-1] + k
        
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