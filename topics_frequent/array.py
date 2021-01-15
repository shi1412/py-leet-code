from utilities.node import ListNode
from utilities.tree import TreeNode
from collections import deque, defaultdict
import heapq

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