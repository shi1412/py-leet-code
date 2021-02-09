from utilities.node import ListNode
from utilities.tree import TreeNode
from collections import deque, defaultdict
import heapq

class two_sum_1:
    def twoSum(self, nums, target):
        obj = {}
        for i in range(len(nums)):
            if target - nums[i] in obj:
                return [i, obj[target - nums[i]]]
            
            obj[nums[i]] = i

class add_two_numbers_2:
    """
    time: O(n)
    space: O(n)
    """
    def addTwoNumbers(self, l1, l2):
        dummy = ListNode(0)
        sum_ = 0
        cur = dummy
        p1, p2 = l1, l2
        while p1 or p2:
            if p1:
                sum_ += p1.val
                p1 = p1.next
                
            if p2:
                sum_ += p2.val
                p2 = p2.next
                
            cur.next = ListNode(sum_ % 10)
            sum_ //= 10
            cur = cur.next
            
        if sum_ == 1:
            cur.next = ListNode(1)
            
        return dummy.next

class container_with_most_water_11:
    def maxArea(self, height):
        if height is None or len(height) == 0:
            return 0 
        
        left  = 0
        right = len(height) - 1
        res = 0
        while left < right:
            res = max(res, min(height[left], height[right]) * (right - left))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
                
        return res
    
class remove_duplicates_from_sorted_array_26:
    def removeDuplicates(self, nums):
        if nums is None or len(nums) == 0:
            return 0

        count  = 0
        for i in range(len(nums)):
            if nums[i] != nums[count]:
                count += 1
                nums[count] = nums[i]
                
        return count + 1

class remove_element_27:
    def removeElement(self, nums, val):
        if nums is None or len(nums) == 0:
            return 0
        
        res = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[res] = val
                res += 1
        
        return res                

class find_first_and_last_postion_of_element_in_sorted_array_34:
    def searchRange(self, nums, target):
        res  = [-1, -1]
        for i in range(len(nums)):
            if nums[i] == target:
                res[0] = i
                break
            
        if res[0] == -1:
            return res
        
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] == target:
                res[1] = i
                break
        
        return res

class valid_sudoku_36:
    def isValidSudoku(self, board):
        if board is None or len(board) == 0:
            return False
        
        for i in range(len(board)):
            rows = set()
            cols = set()
            cube = set()
            for j in range(len(board[0])):
                if board[i][j] != "." and board[i][j] in rows:
                    return False
                
                rows.add(board[i][j])
                if board[j][i] != "." and board[j][i] in cols:
                    return False
                
                cols.add(board[j][i])
                
                row_index = 3 * (i // 3)
                col_index = 3 * (i % 3)
                if board[row_index + j // 3][col_index + j % 3] != "." and board[row_index + j // 3][col_index + j % 3] in cube:
                    return False
                
                cube.add(board[row_index + j // 3][col_index + j % 3])
                
        return True

class sudoku_solver_37:
    def solveSudoku(self, board):
        if board is None or len(board) == 0:
            return 
        
        self.solve(board)
        return board
    
    def solve(self, board):
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == ".":
                    for c in range(1, 10):
                        if self.isValid(board, i, j, str(c)):
                            board[i][j] = str(c)
                            if self.solve(board):
                                return True
                            else:
                                board[i][j] = "."
                    
                    return False
        
        return True
    
    def isValid(self, board, row, col, c):
        for i in range(9):
            if board[i][col] != "." and board[i][col] == c:
                return False
            
            if board[row][i] != "." and board[row][i] == c:
                return False
            
            row_index = 3 * (row // 3)
            col_index = 3 * (col // 3)
            if board[row_index + i // 3][col_index + i % 3] != "." and board[row_index + i // 3][col_index + i % 3] == c:
                return False
            
        return True
    
class trapping_rain_water_42:
    def trap(self, height):
        if height is None or len(height) == 0:
            return 0
        
        left = 0
        right = len(height) - 1
        res = 0
        left_max = 0
        right_max = 0
        while left < right:
            if height[left] < height[right]:
                left_max = max(height[left], left_max)
                res += left_max - height[left]
                left += 1
            else:
                right_max = max(height[right], right_max)
                res += right_max - height[right]
                right -= 1
                
        return res

class rotate_image_48:
    def rotate(self, matrix):
        n = len(matrix)
        for i in range(n):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
                
        for i in range(n):
            for j in range(n // 2):
                matrix[i][j], matrix[i][n - 1- j] = matrix[i][n - 1 -j], matrix[i][j]
                
        return matrix
             
class maximum_subarray_53:
    def maxSubArray_A(self, nums):
        if nums is None or len(nums) == 0:
            return 0
        
        dp = [0]*len(nums)
        dp[0] = nums[0]
        res = nums[0]
        for i in range(1, len(nums)):
            if dp[i - 1] < 0:
                dp[i] = nums[i]
            else:
                dp[i] = nums[i] + dp[i - 1]
            
            res = max(res, dp[i])
            
        return res
    
    def maxSubArray_B(self, nums):
        __sum = nums[0]
        res = nums[0]
        for i in range(1, len(nums)):
            __sum = max(nums[i], nums[i] + __sum)
            res = max(res, __sum)
            
        return res   

class spiral_matrix_54:
    def spiralOrder(self, matrix):
        res = []
        if matrix is None or len(matrix) == 0 or len(matrix[0]) == 0:
            return res
        
        col_begin = 0
        col_end = len(matrix[0]) - 1
        row_begin = 0
        row_end = len(matrix) - 1
        while row_begin <= row_end and col_begin <= col_end:
            for i in range(col_begin, col_end + 1):
                res.append(matrix[row_begin][i])
                
            row_begin += 1
            for i in range(row_begin, row_end + 1):
                res.append(matrix[i][col_end])
                
            col_end -= 1
            if row_begin <= row_end:
                for i in range(col_end, col_begin - 1 , -1):
                    res.append(matrix[row_end][i])
                    
            row_end -= 1
            if col_begin <= col_end:
                for i in range(row_end, row_begin - 1, -1):
                    res.append(matrix[i][col_begin])
                    
            col_begin += 1
            
        return res
           
class merge_intervals_56:
    def merge(self, intervals):
        if intervals is None or len(intervals) == 0:
            return intervals
        
        intervals.sort(key=lambda x:x[0])
        start = intervals[0][0]
        end = intervals[0][1]
        res = []
        for interval in intervals:
            if interval[0] <= end:
                end = max(end, interval[1])
            else:
                # Before update the start and end
                # append the last result to the final array
                res.append([start, end])
                start = interval[0]
                end = interval[1]
                
        res.append([start, end])
        return res

class insert_interval_57:
    def insert(self, intervals, newInterval):
        if newInterval is None:
            return intervals
        res = []
        i = 0
        while i < len(intervals) and intervals[i].end < newInterval[0]:
            res.append(intervals[i])
            i += 1
            
        while i < len(intervals) and intervals[i].start <= newInterval[1]:
            newInterval[0] = min(newInterval[0], intervals[i][0])
            newInterval[1] = max(newInterval[1], intervals[i][1])
            i += 1
            
        res.append(newInterval)
        while i < len(intervals):
            res.append(intervals[i])
            i += 1
            
        return res 

class spiral_matrix_II_59:
    def generateMatrix(self, n):
        res = [[0]*n for _ in range(n)]
        row_begin = 0
        row_end = n - 1
        col_begin = 0
        col_end = n - 1
        num = 1
        while row_begin <= row_end and col_begin <= col_end:
            for i in range(col_begin, col_end + 1):
                res[row_begin][i] = num
                num += 1
                
            row_begin += 1
            for i in range(row_begin, row_end + 1):
                res[i][col_end] = num
                num += 1
                
            col_end -= 1
            for i in range(col_end, col_begin - 1, -1):
                res[row_end][i] = num
                num += 1
                
            row_end -= 1
            for i in range(row_end, row_begin - 1, -1):
                res[i][col_begin] = num
                num += 1
                
            col_begin += 1
        
        return res

class set_matrix_zeroes_73:
    def setZeroes(self, matrix):
        if matrix is None or len(matrix) == 0 or len(matrix[0]) == 0:
            return 0
        
        row_num = len(matrix)
        col_num = len(matrix[0])
        row = False
        col = False
        
        for i in range(row_num):
            for j in range(col_num):
                if matrix[i][j] == 0:
                    matrix[0][j] = 0
                    matrix[i][0] = 0
                    if i == 0:
                        row = True
                        
                    if j == 0:
                        col = True
        
        for i in range(1, row_num):
            if matrix[i][0] == 0:
                for j in range(1, col_num):
                    matrix[i][j] = 0
                    
        for j in range(1, col_num):
            if matrix[0][j] == 0:
                for i in range(1, row_num):
                    matrix[i][j] = 0
                    
        if row:
            for j in range(col_num):
                matrix[0][j] = 0
                
        if col:
            for i in range(row_num):
                matrix[i][0] = 0
                
        return matrix
        
class sort_colors_75:
    """ Dutch National Flag Problem
    time: O(n)
    space: O(1)
    """
    def sortColors(self, nums):
        if nums is None or len(nums) == 0:
            return
        
        left = 0 # define the boundary for 0
        right = len(nums) - 1 # define the boundary for 1
        curr = 0
        
        while  curr <= right:
            if nums[curr] == 0: 
                # While the current value is 0, then swap with left and move both pointers 
                nums[curr], nums[left] = nums[left], nums[curr]
                curr += 1
                left += 1
            elif nums[curr] == 2:
                # while the current value is 2, then swap with right and move right pointers to left
                nums[curr], nums[right] = nums[right], nums[curr]
                right -= 1
            else:
                # while the current value is 1, then just move the pointer to the right
                curr += 1

class remove_duplicates_from_sorted_array_80:
    def removeDuplicates(self, nums): 
        if len(nums) <= 2:
            return len(nums)
        
        count = 2
        for i in range(2, len(nums)):
            if nums[i] != nums[count - 2]:
                nums[count] = nums[i]
                count +=1 
                
        return count
          
class merge_sorted_array_88:
    def merge(self, nums1, nums2, m, n):
        if nums1 is None or len(nums1) == 0:
            return nums2
        
        if nums2 is None or len(nums2) == 0:
            return nums1
        
        i = m - 1
        j = n - 1
        k = m + n -1
        while i >= 0 or j >= 0:
            if nums1[i] >= nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
                
            k -= 1
            
        while j >= 0:
            nums1[k] = nums2[j]
            j -= 1
            k -= 1
            
        return nums1

class gas_station_134:
    def canCompleteCircuit(self, gas, cost):
        if gas is None or cost is None or len(gas) == 0 or len(cost) == 0 or len(gas) < len(cost):
            return -1
        
        total, sum_, start = 0, 0, 0
        for i in range(len(gas)):
            total += gas[i] - cost[i]
            if sum_ < 0:
                sum_ = gas[i] - cost[i]
                start = i
            else:
                sum_ += gas[i] - cost[i]
        
        return -1 if total < 0 else start

class maximum_product_subarray_152:
    def maxProduct(self, nums):
        if nums is None or len(nums) == 0:
            return 0
        
        __max = nums[0]
        __min = nums[0]
        res = nums[0]
        for i in range(1, len(nums)):
            temp = __max
            __max = max(__max * nums[i], __min * nums[i], nums[i])
            __min = min(temp * nums[i], __min * nums[i], nums[i])
            res = max(res, __max)
            
        return res
           
class missing_range_163:
    def findMissingRanges(self, nums, lower, upper):
        res = []
        if nums is None or len(nums) == 0:
            res.append(self._format_range(lower, upper))
            return res
        
        if nums[0] > lower:
            res.append(self._format_range(lower, nums[0] - 1))
            
        for i in range(1, len(nums)):
            if nums[i] - nums[i - 1] > 1:
                res.append(self._format_range(nums[i - 1] + 1, nums[i] - 1))
                
        if nums[-1] < upper:
            res.append(self._format_range(nums[-1] + 1, upper)) 
       
        return res 
   
    @staticmethod    
    def _format_range(lower, upper):
        if lower == upper:
            return str(lower)
        else:
            return "{0}->{1}".format(lower, upper)

class rotate_array_189:
    def rotate(self, nums, k):
        if nums is None or len(nums) == 0:
            return nums
        
        k = k % len(nums)
        nums[:k], nums[k:] = nums[len(nums) - k:], nums[:len(nums) - k]
        return nums

class minimum_size_subarray_sum_209:
    def minSubArray(self, s, nums):
        res = float("inf")
        left, __sum= 0, 0 
        for i in range(len(nums)):
            __sum += nums[i]
            while left < i and __sum >= s:
                res = min(res, i - left + 1)
                __sum -= nums[left]
                left += 1
                
        return res if res != float("inf") else 0
    
class kth_largest_element_in_array_215:
    def findKthLargest(self, nums, k):
        if nums is None or len(nums) == 0:
            return 0
        
        priority_queue = []
        for i in nums:
            heapq.heappush(priority_queue, i)
            if len(nums) > k:
                heapq.heappop(priority_queue)
                
        return heapq.heappop(priority_queue)

class summary_ranges_228:
    def summaryRanges(self, nums):
        if nums is None or len(nums) == 0:
            return nums
        
        res = []
        i = 0
        while i < len(nums):
            num = nums[i]
            while i < len(nums) - 1 and nums[i] + 1 == nums[i + 1]:
                i += 1
                
            if num != nums[i]:
                res.append("{0}->{1}".format(num, nums[i]))
            else:
                res.append(str(num))
                
            i += 1
        
        return res

class product_of_array_except_self_238:
    def productExceptSelf(self, nums):
        if nums is None or len(nums) == 0:
            return nums
        
        res = [1] + [0]*(len(nums) - 1)
        # calculate the product except self for the left
        for i in range(1, len(nums)):
            res[i] = res[i - 1] * nums[i - 1]
        
        right = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= right
            right *= nums[i]
            
        return res
                 
class shortest_word_distance_243:
    def shortestDistance(self, words, word1, word2):
        if words is None or len(words) == 0:
            return -1
        
        if word1 is None or word2 is None:
            return -1
        
        res = len(words)
        a = -1
        b = -1
        for i in range(len(words)):
            if words[i] == word1:
                a = i
            elif words[i] == word2:
                b = i
            
            if a != -1 and b != -1:
                res = min(res, abs(a-b))
                
        return res

class shortest_word_distance_244:
    def __init__(self, words):
        self.__map = defaultdict(list)
        for index, elem in enumerate(words):
            self.__map[elem].append(index)
            
    def shortest(self, word1, word2):
        list1, list2 = self.__map[word1], self.__map[word2]
        pos1, pos2 = 0, 0
        res = float('inf')
        
        while pos1 < len(list1) and pos2 < len(list2):
            res = min(res, abs(list1[pos1] - list2[pos2]))
            if list1[pos1] < list2[pos2]:
                pos1 += 1
            else:
                pos2 += 1
                
        return res

class shortest_word_distance_245:
    def shortestDistance(self, words, word1, word2):
        a = -1
        b = -1
        res = len(words)
        for i in range(len(words)):
            if words[i] == word1:
                a = i
                
            if words[i] == word2:
                if word1 == word2:
                    a = b
                    
                b = i
                
            if a != -1 and b != -1:
                res = min(res, abs(a - b))
                
        return res 

class meeting_rooms_252:
    def canAttendMeetings(self, intervals):
        if intervals is None or len(intervals) == 0:
            return None
        
        intervals.sort(key=lambda x: x[0])
        for i in range(1, len(intervals)):
            if intervals[i-1][1] > intervals[i][0]:
                return False
            
        return True
    
class meeting_rooms_253:
    def minMeetingRooms_A(self, intervals):
        if intervals is None or len(intervals) == 0:
            return 0
        
        starts = [interval[0] for interval in intervals]
        ends = [interval[1] for interval in intervals]
        
        starts.sort()
        ends.sort()
        res = 0
        end = 0
        for i in range(len(intervals)):
            if starts[i] < ends[end]:
                res += 1
            else:
                end +=1
                
        return res
    
    def minMeetingRooms_B(self, intervals):
        """
            res[0]
        |___|
               |________|   => intervals[1][0] > res[0]
        intervals[1][0]
        """
        if intervals is None or len(intervals) == 0:
            return 0
        
        intervals.sort(key=lambda x:x[0])
        res = []
        heapq.heappush(res, intervals[0][1])
        for i in range(1, len(intervals)):
            if intervals[i][0] >= res[0]:
                heapq.heappop(res)
                
            heapq.heappush(res, intervals[i][1])
            
        return len(res)

class h_index_274:
    def hIndex_A(self, citations):
        """
        T: O(nlogn)
        S: O(1)
        """
        citations.sort()
        res = 0
        n = len(citations)
        while res < n and citations[n - 1 - res] > res:
            res += 1
            
        return res
    
class wiggle_sort_280:
    def wiggleSort(self, nums):
        if nums is None or len(nums) == 0:
            return
        
        for i in range(1, len(nums)):
            if (i % 2 == 1 and nums[i] < nums[i - 1]) or (i % 2 == 0 and nums[i] > nums[i - 1]):
                nums[i], nums[i - 1] = nums[i - 1], nums[i] 
                
        return nums
           
class move_zeroes_283:
    def moveZeroes(self, nums):
        if nums is None or len(nums) == 0:
            return nums
        
        cur = 0
        for i in range(len(nums)):
            if nums[i]:
                nums[cur] = nums[i]
                
        for i in range(cur, len(nums)):
            nums[i] = 0
            
class find_the_duplicate_number_287:
    def findDuplicate(self, nums):
        if nums is None or len(nums) == 0:
            return nums
        
        dup = set()
        for num in nums:
            if num in dup:
                return num
            
            dup.add(num)

class game_of_life_289:
    def gameOfLife_A(self, board):
        if board is None or len(board) == 0 or len(board[0]) == 0:
            return board
        
        neighbors = [(0, -1), (0, 1), \
            (-1, -1), (-1, 0), (-1, 1), \
            (1, -1), (1, 0), (1, 1)]
        
        row = len(board)
        col = len(board[0])
        dup_board = [[board[i][j] for j in range(col)] for i in range(row)]
        
        for i in range(row):
            for j in range(col):
                count = 0
                for elem in neighbors:
                    r = i + elem[0]
                    c = j + elem[1]
                    if (r < row and r >= 0) and (c < col and c >=0) and dup_board[r][c] == 1:
                        count += 1
                        
                if dup_board[i][j] == 1 and (count < 2 or count > 3):
                    board[i][j] = 0
                    
                if dup_board[i][j] == 0 and count == 3:
                    board[i][j] = 1
                    
        return board
    
class longest_increasing_subsequence_300:
    def lengthOfLIS_A(self, nums):
        """
        1. use the idea of insertion sort to restructure the array into a new array
        2. use the binary search methodology to find the next number that will make the list increasing
        """
        # Define an empty array as a placeholder for the new array
        container = [0]*len(nums)
        # Define a global variable to hold the final result
        res = 0
        for num in nums:
            i, j = 0, res
            while i != j:
                mid = (i + j) // 2
                if container[mid] < num:
                    i = mid + 1
                else:
                    j = mid
            
            container[i] = num
            if res == i:
                res += 1
                
        return res
    
    def lengthOfLIS_B(self, nums):
        """using dynamic programming
        status:  example: [10, 9, 2, 5, 3, 7, 101, 18] -> status metric: [1, 1, 1, 2, 2, 3, 4, 4]
        """
        if nums is None or len(nums) == 0:
            return 0
        
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            for j in range(0, i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
                    
        res = dp[0]
        for i in range(1, len(dp)):
            res = max(res, dp[i])
            
        return res

class sparse_matrix_multiplicatoin_311:
    def multiply(self, A, B):
        if A is None or B is None:
            return [[]]
        
        if len(A) == 0 or len(B) == 0:
            return [[]]
        
        a_row, b_row, b_col = len(A), len(B), len(B[0])
        res = [[0]*b_col for _ in range(a_row)]
        for i in range(a_row):
            for j in range(b_row):
                if A[i][j] != 0:
                    for k in range(b_col):
                        if B[j][k] != 0:
                            res[i][k] += A[i][j] * B[j][k]
                            
        return res
        
class maximum_size_subarray_sum_equals_k_325:
    def maxSubArrayLen(self, nums, k):
        obj = {0: -1}
        cur  = 0
        res = 0
        for i in range(len(nums)):
            cur += nums[i]
            if cur - k in obj:
                res = max(res, i - obj[cur - k])
                
            if cur not in obj:
                obj[cur] = i
        
        return res        
                   
class increasing_triplet_subsequence_334:
    def increasingTriplet(self, nums):
        if nums is None or len(nums) == 0:
            return False
        
        first = float('inf')
        second = float('inf')
        
        for n in nums:
            if n <= first:
                first = n
            elif n <= second:
                second = n
            else: 
                return True
            
        return False

class range_addition_370:
    def getModifiedArray(self, length, updates):
        res = [0]*(length + 1)
        for i, j, d in updates:
            res[i] += d
            res[j + 1] -= d
        
        for i in range(1, length):
            res[i] += res[i - 1]
            
        res.pop()
        return res

class wiggle_subsequence_376:
    def wiggleMaxLength(self, nums):
        n = len(nums)
        if n < 2:
            return n
        
        up = [1] + [0]*(n - 1)
        down = [1] + [0]*(n - 1)
        for i in range(1, n):
            if nums[i] > nums[i - 1]:
                up[i] = max(up[i - 1], down[i - 1] + 1)
                down[i] = down[i - 1]
            elif nums[i] < nums[i - 1]:
                up[i] = up[i - 1]
                down[i] = max(up[i - 1] + 1, down[i - 1])
            else:
                up[i] = up[i - 1]
                down[i] = down[i - 1]
                
        return max(up[n - 1], down[n - 1])
    
class first_unique_character_in_a_string_387:
    def firstUniqChar(self, s):
        if s is None or len(s) == 0:
            return -1
        
        counter = {}
        for i in s:
            if i in counter:
                counter += 1
            else:
                counter = 1
                
        for index, num in enumerate(s):
            if counter[num] == 1:
                return index
            
        return -1
            
            
