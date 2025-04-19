class MyQueue:
    def __init__(self):
        # Stack for pushing elements (newest elements)
        self.stack_push = []
        # Stack for popping elements (oldest elements)
        self.stack_pop = []

    def push(self, x: int) -> None:
        # Always push to the push stack
        self.stack_push.append(x)

    def pop(self) -> int:
        # If pop stack is empty, transfer all elements from push stack
        self._transfer_if_needed()
        # Pop from the pop stack (which has elements in FIFO order)
        return self.stack_pop.pop()

    def peek(self) -> int:
        # If pop stack is empty, transfer all elements from push stack
        self._transfer_if_needed()
        # Peek from the pop stack
        return self.stack_pop[-1]

    def empty(self) -> bool:
        # Queue is empty if both stacks are empty
        return len(self.stack_push) == 0 and len(self.stack_pop) == 0
    
    def _transfer_if_needed(self) -> None:
        # Only transfer if pop stack is empty (amortized operation)
        if not self.stack_pop:
            # Transfer all elements from push stack to pop stack
            # This reverses the order, making the oldest element on top
            while self.stack_push:
                self.stack_pop.append(self.stack_push.pop())