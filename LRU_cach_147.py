class ListNode:
    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.hashmap = {}
        self.head = ListNode()
        self.tail = ListNode()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def get(self, key: int) -> int:
        # 如果已经在链表中了就把它移到头部（变成最新访问的）
        if key in self.hashmap:
            self.move_node_to_header(key)
        res = self.hashmap.get(key, -1)
        if res == -1:
            return res
        else:
            return res.value

    def put(self, key: int, value: int) -> None:
        if key in self.hashmap:
            # 如果key本身已经在哈希表中了就不需要在链表中加入新的节点
            # 但是需要更新字典该值对应节点的value
            self.hashmap[key].value = value
            # 之后将该节点移到链表头部
            self.move_node_to_header(key)
        else:
            if len(self.hashmap) >= self.capacity:
            # 若cache容量已满，删除cache中最不常用的节点 
                self.pop_tail()
            self.add_node_to_header(key,value)

    def move_node_to_header(self, key):
            # 先将哈希表key指向的节点拎出来
            node = self.hashmap[key]
            node.prev.next = node.next
            node.next.prev = node.prev
            # 将node插入到头部节点前
            node.prev = self.head
            node.next = self.head.next
            self.head.next.prev = node
            self.head.next = node
            
    def add_node_to_header(self, key,value):
        new = ListNode(key, value)
        self.hashmap[key] = new
        new.prev = self.head
        new.next = self.head.next
        self.head.next.prev = new
        self.head.next = new
        
    def pop_tail(self):
        last_node = self.tail.prev
        # 去掉链表尾部的节点在哈希表的对应项
        self.hashmap.pop(last_node.key)
        # 去掉最久没有被访问过的节点，即尾部Tail之前的一个节点
        last_node.prev.next = self.tail
        self.tail.prev = last_node.prev
        return last_node
