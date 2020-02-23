#   Created by Michael Shi on 06/24/18.
#   Copyright © 2018 Michael Shi. All rights reserved.

class Solution:
    """
    @param: number: 3-digit number.
    @return: reversed number
    """
    def reverseInteger(self, number):
        numberStr = ''
        if str(number)[0] == '-':
            numberStr = str(number)[0] + str(numberStr)[1:][::-1]
        else:
            numberStr = str(numberStr)[::-1]
          
        return int(numberStr)