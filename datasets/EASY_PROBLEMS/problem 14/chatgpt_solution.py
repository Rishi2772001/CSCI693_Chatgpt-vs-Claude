class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def deleteDuplicates(head: ListNode) -> ListNode:
    curr = head
    while curr:
        # Skip over any nodes that have the same value as the current one.
        while curr.next and curr.next.val == curr.val:
            curr.next = curr.next.next
        curr = curr.next
    return head

# Helper function to build a linked list from a Python list.
def build_linked_list(lst):
    if not lst:
        return None
    head = ListNode(lst[0])
    curr = head
    for val in lst[1:]:
        curr.next = ListNode(val)
        curr = curr.next
    return head

# Helper function to convert a linked list back to a Python list.
def linked_list_to_list(head):
    result = []
    curr = head
    while curr:
        result.append(curr.val)
        curr = curr.next
    return result
