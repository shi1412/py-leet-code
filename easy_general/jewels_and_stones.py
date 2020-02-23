#   Created by Michael Shi on 06/16/18.
#   Copyright © 2018 Michael Shi. All rights reserved.

#------------------ Question ----------------------
# You're given strings J representing the types of stones that are jewels, and S representing the stones you have.  
# Each character in S is a type of stone you have.  
# You want to know how many of the stones you have are also jewels.
# The letters in J are guaranteed distinct, and all characters in J and S are letters. 
# Letters are case sensitive, so "a" is considered a different type of stone from "A".

#------------------- Note ----------------------
# S and J will consist of letters and have length at most 50.
# The characters in J are distinct.

#------------------- Answer ----------------------

class Solution(object):
    def numJewelsInStones(self, J, S):
        """
        :type J: str
        :type S: str
        :rtype: int
        """
        jew  = list(J)  # convert string to array [a, A]
        stone = list(S) # convert string to array [a, A, A, b, b]
        sumNum = 0
        for i in jew:
             for j in stone:
                    if i == j:
                        sumNum = sumNum + 1
                    else:
                        sumNum = sumNum + 0
        
        return sumNum
