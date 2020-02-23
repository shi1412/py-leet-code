#   Created by Michael Shi on 06/20/18.
#   Copyright © 2018 Michael Shi. All rights reserved.

#------------------ Question ----------------------
# International Morse Code defines a standard encoding where each letter is mapped to a series of dots and dashes, 
# as follows: "a" maps to ".-", "b" maps to "-...", "c" maps to "-.-.", and so on.
# Now, given a list of words, each word can be written as a concatenation of the Morse code of each letter. 
# For example, "cab" can be written as "-.-.-....-", (which is the concatenation "-.-." + "-..." + ".-"). 
# We'll call such a concatenation, the transformation of a word.
# Return the number of different transformations among all words we have.

#------------------- Note ----------------------
# The length of words will be at most 100.
# Each words[i] will have length in range [1, 12].
# words[i] will only consist of lowercase letters.

#------------------- Answer ----------------------

class Solution(object):
    def uniqueMorseRepresentations(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        morseArr = []

    def helpFunction(self, words):

        letterArr = ['a','b','c','d','e','f','g','h','i','j','k','l',
                     'm','n','o','p','q','r','s','t','u','v','w','x','y','z']
        morseStr = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---",
                    "-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-",
                    "...-",".--","-..-","-.--","--.."]
        
        self.arr = list(words)
        for i in self.arr:
            self.arr[i] = morseStr[letterArr.index(i)]

        return ''.join(self.arr)