from collections import deque

class TreeNode:
    """ Definition of a binary tree node."""
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
 
tn = TreeNode(4)
tn.left = TreeNode(2)
tn.right = TreeNode(6) 
tn.left.left = TreeNode(1)
tn.left.right = TreeNode(3)
tn.right.left = TreeNode(5)
tn.right.right = TreeNode(7)
       
# --------------------------
# Preorder 
# LeetCode 114
# --------------------------
class Preorder:
    def __init__(self):
        self.res = []
        
    def recursion(self, root):
        if root is None:
            return self.res
        
        self.helper(root)
        return self.res
    
    def helper(self, root):
        if root is None:
            return
        
        self.res.append(root.val)
        self.helper(root.left)
        self.helper(root.right)    
        
    def iteration(self, root):
        if root is None:
            return self.res
        
        stack = [root]
        while stack:
            cur = stack.pop()
            self.res.append(cur.val)
            if cur.right:
                stack.append(cur.right)
            
            if cur.left:
                stack.append(cur.left)
        
        return self.res
# --------------------------
# Inorder
# LeetCode 94
# --------- -----------------
class Inorder:
    def __init__(self):
        self.res = []
        
    def recursion(self, root):
        """Recursion traverse
        """
        if root is None:
            return self.res
        
        self.helper(root)
        return self.res
        
    def helper(self, root):
        if root is None:
            return
        
        self.helper(root.left)
        self.res.append(root.val)
        self.helper(root.right)
        
    def iteration(self, root):
        """Iteration traverse
        """
        if root is None:
            return self.res
        
        stack = []
        while root or len(stack) > 0:
            while root:
                stack.append(root)
                root = root.left
            
            root = stack.pop()
            self.res.append(root.val)
            root = root.right
            
        return self.res   
# --------------------------
# Postorder 
# LeetCode 145
# --------------------------
class Postorder:
    def __init__(self):
        self.res = []
        
    def recursion(self, root):
        if root is None:
            return self.res
        
        self.helper(root)
        return self.res
        
    def helper(self, root):
        if root is None:
            return
        
        self.helper(root.left)
        self.helper(root.right)
        self.res.append(root.val)

    def iteration(self, root):
        if root is None:
            return self.res
        
        stack = []
        while root or stack:
            while root:
                if root.right:
                    stack.append(root.right)
                
                stack.append(root)
                root = root.left
                
            root = stack.pop()
            if stack and root.right == stack[-1]:
                stack[-1] = root
                root = root.right
            else:
                self.res.append(root)
                root = None
                
        return self.res
        
# --------------------------
# Levelorder 
# --------------------------
class LevelOrder:
    def __init__(self):
        self.res = []
        
    def recursion(self, root):
        if root is None:
            return self.res
        
        self.helper(root, 0)
        return self.res
    
    def helper(self, root, level):
        if root is None:
            return
        
        if level == len(self.res):
            self.res.append([])
        
        self.res[level].append(root.val)
        if root.left:
            self.helper(root.left, level + 1)
            
        if root.right:
            self.helper(root.right, level + 1)
            
    def iteration(self, root):
        if root is None:
            return self.res
        
        queue = deque([root, ])
        while queue:
            size = len(queue)
            i = 0
            while i <= size:
                cur = queue.popleft()
                if cur.left:
                    queue.append(cur.left)
                    
                if cur.right:
                    queue.append(cur.right)
                
                i += 1
                self.res.append(cur.val)
                
        return self.res
        