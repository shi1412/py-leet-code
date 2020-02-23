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
    
     