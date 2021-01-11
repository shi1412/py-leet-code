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