# 1 Two Sum
def twoSum(nums, target):
    dic = {}
    for i in range(nums):
        pair = target - nums[i]
        if pair in dic:
            return [dic[pair], i]
        else:
            dic[nums[i]] = i
    return None

# 8 String to Integer (atoi)
def myAtoi(strs):
    # remove the white space
    new_str = strs.lstrip()
    if new_str is None or len(new_str) == 0:
        return 0
    
    sign = 1
    start = 0
    result = 0
    if new_str[0] == "+":
        start += 1
        sign = 1
        
    if new_str[0] == "-":
        start += 1
        sign = -1
        
    for i in range(start, len(new_str)):
        if new_str[i].isnumeric() == False:
            return int(result)* sign
        
        result = result * 10 + int(new_str)
        
        if sign == 1 and result > 2147483647:
            return 2147484647
        
        if sign == -1 and result < -2147483648:
            return -2147483648         
    
    return int(result)*sign

# 49 Group Anagrams
# dictionary
# string sorting
def groupAnagrams(strs):
        group = {}
        if len(strs) < 1:
            return strs

        for i in range(len(strs)):
            elem = strs[i]
            elem_sort = "".join(sorted(elem))
            if elem_sort in group:
                group[elem_sort].append(elem)
            else:
                group[elem_sort] = [elem]

        return group.values()

# 46 Permutations
def permute(nums):
    if len(nums) <=1:
        return [nums]

    ans = []
    for index, num in enumerate(nums):
        n = nums[:index] + nums[index+1:]
        for y in permute(n):
            ans.append([num] + y)

    return ans

# 118. Pascal's Triangle
def generate(numRows):
    if numRows is None:
        return []
    
    result = []
    for i in range(numRows):
        result.append([])
        for j in range(i+1):
            if j in (0, i):
                result[i].append(1)
            else:
                result[i].append(result[i-1][j] + result[i-1][j-1])
                
    return result
       
# 119. Pascal's Triangle II  
def getRow(rowIndex):
    if rowIndex is None:
        return []
    
    result = [1] + [0]*rowIndex
    for i in range(rowIndex):
        result[0] = 1
        for j in range(i+1, 0, -1):
            result[j] = result[j-1] + result[j]
            
    return result
     
# 166. Fraction to Recurring Decimal
def fractionToDecimal(numerator, denominator):
    # consider the numerator is equal to 0 case
    if numerator == 0:
        return "0"
    # consider the sign
    result = "-" if ((numerator >0 ) ^ (denominator >0)) else ""
    # remove the sign
    num = abs(numerator)
    den = abs(denominator)
    # use the build in function divmod to get the quotient and remainder
    quo, rem = divmod(num, den)
    if rem == 0: return result + str(quo)
    result += str(quo) + "."
    # define a dictionary to record if the remainder is repeat
    pos = {}
    pos[rem] = len(result)
    while rem != 0:
        rem *= 10
        quo, rem = divmod(rem, den)
        result += str(quo)
        if rem in pos:
            index = pos[rem]
            result = result[index:] + "(" + result[index:] + ")"
            break
        else:
            pos[rem] = len(result)

    return result

# 238. Product of Array Except Self
def productExceptSelf(nums):
    # the idea is to mulitple each element from left to right
    # and then from right to left
    # since you can't time itself, the loop will start from 1
    if nums is None or len(nums) == 0:
        return nums

    result = [1] + [0]* (len(nums)-1)
    for i in range(1, len(nums)):
        result[i] = result[i-1]* nums[i-1]

    right = 1
    for i in range(len(nums)-1, -1, -1):
        result[i] *= right
        right *= nums[i]

    return result

# 387 First Unique Character in a String
def firstUniqChar(s):
    counter = {}
    for i in range(len(s)):
        if i in counter:
            counter[i] += 1
        else:
            counter[i] = 1
            
    for index, elem in enumerate(s):
        if counter[elem] == 1:
            return index
        
    return -1
# 74 Search a 2D Matrix
# think like a binary search
 def searchMatrix(matrix, target):
     if matrix is None or len(matrix) == 0:
         return False
     
     row = len(matrix)
     col = len(matrix[0])
     start = 0
     end = row * col - 1
     while start <= end:
         mid = (end - start)//2 + start
         mid_value = matrix[mid//col][mid%col]
         if mid_value == target:
             return True
         elif mid_value < target:
             start = mid + 1
         else:
             end = mid - 1
             
    return False

# 1086 High Five
def highFive(items):
    if len(items) == 0 or len(items[0]) == 0:
        return []

    score = {}
    for item in items:
        _id = item[0]
        _score = item[1]
        if _id in score:
            score[_id].append(_score)
        else:
            score[_id] = [_score]

    result = []
    for k in score:
        temp = score[k]
        temp_sort = temp.sort()
        result.append(sum(temp_sort[-5:])//5)

    return result