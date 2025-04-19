class MinStack:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack = []  # Main stack to store values
        self.min_stack = []  # Auxiliary stack to track minimums
        
    def push(self, val: int) -> None:
        """
        Pushes an element onto the stack.
        """
        self.stack.append(val)
        
        # If min_stack is empty or val is less than or equal to current minimum,
        # add val to min_stack
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
        
    def pop(self) -> None:
        """
        Removes the element on top of the stack.
        """
        if self.stack:
            # If the popped element is the current minimum, also pop from min_stack
            if self.stack[-1] == self.min_stack[-1]:
                self.min_stack.pop()
            self.stack.pop()
        
    def top(self) -> int:
        """
        Get the top element of the stack.
        """
        if self.stack:
            return self.stack[-1]
        return None
        
    def getMin(self) -> int:
        """
        Retrieve the minimum element in the stack.
        """
        if self.min_stack:
            return self.min_stack[-1]
        return None