class MyQueue:
    def __init__(self):
        # Two stacks: in_stack for incoming elements, out_stack for outgoing elements.
        self.in_stack = []
        self.out_stack = []

    def push(self, x: int) -> None:
        # Simply push to in_stack.
        self.in_stack.append(x)

    def pop(self) -> int:
        # Ensure out_stack has the current queue order.
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        # Pop from out_stack which represents the front of the queue.
        return self.out_stack.pop()

    def peek(self) -> int:
        # Ensure out_stack has the current queue order.
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        # Peek at the top of out_stack, which is the front element.
        return self.out_stack[-1]

    def empty(self) -> bool:
        # The queue is empty if both stacks are empty.
        return not self.in_stack and not self.out_stack