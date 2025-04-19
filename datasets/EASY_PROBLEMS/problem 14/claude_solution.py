# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

def deleteDuplicates(head):
    # Edge case: empty list or single node
    if not head or not head.next:
        return head
    
    current = head
    
    # Traverse the list
    while current and current.next:
        # If current node value equals next node value (duplicate found)
        if current.val == current.next.val:
            # Skip the next node by pointing to the node after it
            current.next = current.next.next
        else:
            # Move to the next node if no duplicate
            current = current.next
    
    return head