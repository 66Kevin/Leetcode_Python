[TOC]

# Leetcode刷题笔记

## 一. 位运算符

### 1. 两个相同的数字做异或(^), 等于0

- 0与任何数字异或还是该数字本身 
- 例题：136. [只出现一次的数字](https://leetcode-cn.com/problems/single-number) -- 除了一个数字出现一次，其他都出现了两次，让我们找到出现一次的数
- 解法：直接把所有的元素一起异或(^)，剩下的就是只出现一次的数字。

#### [693. 交替位二进制数](https://leetcode-cn.com/problems/binary-number-with-alternating-bits/)

n与n右移的做异或运算，如果n是1和0交替，则异或的结果为都是0

怎么判断异或结果都为0：与n+1相&得到是0

```c++
class Solution {
public:
    bool hasAlternatingBits(int n) {
        n = (n ^ (n >> 1));
        return (n & ((long)n + 1)) == 0; // long为了防止溢出发生
    }
};
```

#### [面试题 05.01. 插入](https://leetcode-cn.com/problems/insert-into-bits-lcci/)

- 我们可以先把数字 N 的第 i ~ j 位区域 清 0, 然后再将 M 左移 i 位, 最后 取或 即可得到最终结果，这里左移和取或都是非常直接的位运算, 关键在于如何清 0。
- 如果我们能够构造出第 i ~ j 位区域为 0 而其他位都为 1 的数字 mask, 那么只需要将它和 N 取交, 即可将这部分区域清 0
  直接构造这个数字不太方便, 我们可以反其道而行之, 先得到第 i ~ j 位区域为 1 而其他位都为 0 的数字 mask', 然后逐位取反 (~) 即可
  这个数字mask'如何得到呢? 注意到这个区域的长度 length 为 j-i+1, 所以我们可以利用 (1<<length) - 1得到从低位到高共计 length 个 1 的数字, 然后左移 i 位即可

```c++
class Solution {
public:
    int insertBits(int N, int M, int i, int j) {
    	int mask = 0;
    	for(int k = i; k <= j; ++k)
    		mask |= (1<<k);//mask 000001110000
    	N &= (~mask);//清零N中间的位, mask 111110001111
    	M <<= i;
    	return N | M;
    }
};
```

```python
class Solution:
    def insertBits(self, N: int, M: int, i: int, j: int) -> int:
        # mask代表[i,j]之间的位为0, 其他位为1的数字
        mask = 0
        for k in range(i, j+1):
            mask |= (1 << k) # mask : 100 + 1000 + 1000 + 10000 -> 11110000
            								 # ~mask : 00001111
              							 # N & ~mask: 1111111111 & 00001111 -> 1100001111
                						 # M << i: M=1011,i=4 -> 10110000
                  					 # (N & ~mask) | (M << i): 11,00001111|10110000 -> 11,10110000
        return (N & ~mask) | (M << i) # N & ~mask将i到j之间的位清0, 然后将M左移i位, 并与N取或, 即得到最终结果
```

#### [剑指 Offer II 005. 单词长度的最大乘积](https://leetcode-cn.com/problems/aseY1I/)Median

暴力解法：把每个词word[i]变成一个set，set中为word[i]的字母，用result保存所有set，一一比较set是否有交集

比如输入`["abcw","baz","foo","bar","xtfn","abcdef"]`

`result:[{'a', 'w', 'c', 'b'}, {'a', 'z', 'b'}, {'o', 'f'}, {'a', 'b', 'r'}, {'x', 'n', 't', 'f'}, {'b', 'a', 'f', 'd', 'e', 'c'}]`

```python
class Solution(object):
    def maxProduct(self, words):
        result =[]
        maxmul = 0
        word_num = len(words)
        for word in words:
            word_set = set()
            for i in range(len(word)):
                word_set.add(word[i]) #把word[i]中每个字母都存入一个set中
            result.append(word_set)
        for i in range(word_num):
            for j in range(i+1,word_num):
                if len(result[i] & result[j])==0: # 两个set求交集，如果是空集说明没有重复字母
                    maxmul = max(maxmul,len(words[i])*len(words[j]))
        return maxmul
```

**位运算解法**:

1byte = 8bit

int = 4 byte = 32 bit

26位字母可以用一个二进制的int来表示，某个字母是否存在

比如，`abcwz`可以表示成`10010000000000000000000111`, `azb`可以表示为`10000000000000000000000011`,将两者&操作，只有重复字母时才会不为0

```python
用一个26位的二进制数字记录每个word[i]中字母的出现情况。
对两个单词对应的二进制数字进行与运算，如果结果为0，说明没有重复。
class Solution(object):
    def maxProduct(self, words):
        result = [0] * len(words)
        maxmul = 0
        for i in range(len(words)):
            for j in words[i]:
                result[i] |= 1 << (ord(j) - 97) # 97是小写a的ascii
        for i in range(len(words)-1):
            for j in range(i+1,len(words)):
                if not (result[i]&result[j]): # 等价于 (result[i] & result[j]) == 0
                    maxmul = max(maxmul,len(words[i])*len(words[j]))
        return maxmul
```

### 2. 位运算反转位

```python
十进制反转位：
  ans = ans * 10 + n % 10 (%取出n的最后一位，加到ans中，/=去掉n的最后一位)
  n /= 10
二进制：
	ans = ans * 2 + ans % 2 (同理)
位运算完成二进制:
  ans = ans << 1 | n & 1 (&取出n的最后一位，加到通过<<扩大一位的ans中，加法操作通过|来实现)
  n >>=1 （去掉n的最后一位）
```
####  190. 32位运算反转:

```c++
class Solution {
public:
    uint32_t reverseBits(uint32_t n) {
        uint32_t res=0;
        for(int i=0; i<32; ++i){
            res = (res << 1) | (n & 1);
            n >>= 1;
        }
        return res;
    }
};
```

#### 191. bit位为1的个数

```python
class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        res = 0
        for _ in range(32):
            tmp = n & 1
            if tmp:
                res +=1
            n>>=1
        return res
```

### 3. 判断一个数是否是2的幂次方

- 2的幂次方写成二进制形式后，二进制中只有一个1，并且1后面跟了n个0；

- ```c++
  例如:8的二进制为1000；8-1=7，7的二进制为111。两者相与的结果为0。计算如下：
      1000
  &   0111
  -------
      0000
  ```

- 因此判断n是否为2的幂可以转化为 ```n &(n-1) == 0```

---

 判断是否是4的幂次方：

将4的幂次方写成二进制形式后发现：二进制中只有一个1（1在奇数位置），并且1后面跟了偶数个0； 因此问题可以转化为判断1后面是否跟了偶数个0就可以了。

       4的整数次幂的二进制数都为 (4)100、(16)10000、(64)1000000......
n为4的幂的前提是n为2的幂，可以用n & (n-1) == 0来判断

4的幂可以转化为$(3+1)^x = 3^x + 1^x = 3^x + 1$

因此可以通过%3来判断是否为4的幂

#### 342. 4的幂

```c++
class Solution {
public:
    bool isPowerOfFour(int n) {
        return n > 0 && (n&(n-1)) == 0 && n % 3 == 1;
    }
};
```

#### [326. 3 的幂](https://leetcode-cn.com/problems/power-of-three/)

x / 3 = y；这个y一定是3的倍数。倍数就用mod来判断` n % 3 == 0`

如果是3的倍数，不断➗3到最后一定剩余1，用`n==1`来结束

```python
class Solution(object):
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        while n and n % 3 == 0:
            n = n / 3
        return n==1
```



### 4. 二进制加法

#### 剑指offer002 二进制加法

思路：逐位相加

需要考虑的点：1. 如果a和b的长度不同怎么办？ 2. 加到最后还剩一个进位值carry怎么办？ 

```python
class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        res = ""
        carry = 0
        i = len(a) - 1
        j = len(b) - 1
        while i >= 0 or j >= 0 or carry != 0:
            digital_A = int(a[i]) if i>=0 else 0  # 如果b还有值，a就补0
            digital_B = int(b[j]) if j>=0 else 0	# 如果a还有值，b就补0
            sum = digital_A + digital_B + carry
            carry = 1 if sum>=2 else 0
            sum = sum - 2 if sum >= 2 else sum
            res += str(sum)
            i-=1
            j-=1
        return res[::-1] # 反转list

```

## 二. 链表问题

1. 对于链表问题，返回结果为头结点时，**通常需要先初始化一个预先指针 pre，该指针的下一个节点指向真正的头结点head**。使用预先指针的目的在于链表初始化时无可用节点值，而且链表构造过程需要指针移动，进而会导致头指针丢失，无法返回结果。

   ```python
   Leetcode add_two_num_2
   class Solution:
       def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
           # 对于链表问题，返回结果为头结点时，通常需要先初始化一个预先指针 pre，该指针的下一个节点指向真正的头结点head。
           # 使用预先指针的目的在于链表初始化时无可用节点值，而且链表构造过程需要指针移动，进而会导致头指针丢失，无法返回结果。
           pre = ListNode(0)
           cur = pre
           remainder = 0 #进位值
   
           while l1 or l2:
               x = l1.val if l1 else 0
               y = l2.val if l2 else 0
               sum = x + y + remainder
   
               remainder = sum//10 #进位值，两个0-9的数相加除10要么0要么1
               sum = sum%10    #实际存入链表中的值，因为只用存个位数即可
   
               cur.next = ListNode(sum)
               cur = cur.next
   
               if l1: l1 = l1.next
               if l2: l2 = l2.next
   
           if remainder == 1:
               cur.next = ListNode(remainder)
   
           return pre.next
   ```

#### [234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)

https://www.youtube.com/watch?v=GTKm1PrYjwo

O(n)和O(1)复杂度，不能存储Node，所以找到中点Node，然后反转中点Node到最后的Node这部分，再检查head到mid和mid到final这部分是否一致即可

1. 找中点node：快慢指针。慢指针一次走一步，快指针一次走两步，快慢指针同时出发。当快指针移动到链表的末尾时，慢指针恰好到链表的中间。通过慢指针将链表分为两部分。若链表有奇数个节点，则中间的节点应该看作是前半部分。
2. 翻转mid到final的list：**previous和current同时后移，并驾齐驱**
3. 最后遍历head到mid和mid到final

```python
class Solution:

    def isPalindrome(self, head: ListNode) -> bool:
        if head is None:
            return True

        # 找到前半部分链表的尾节点并反转后半部分链表
        first_half_end = self.end_of_first_half(head)
        second_half_start = self.reverse_list(first_half_end.next)

        # 判断是否回文
        result = True
        first_position = head
        second_position = second_half_start
        while result and second_position is not None:
            if first_position.val != second_position.val:
                result = False
            first_position = first_position.next
            second_position = second_position.next

        # 还原链表并返回结果
        first_half_end.next = self.reverse_list(second_half_start)
        return result    

    def end_of_first_half(self, head):
        fast = head
        slow = head
        while fast.next is not None and fast.next.next is not None:
            fast = fast.next.next
            slow = slow.next
        return slow

    def reverse_list(self, head):
        previous = None
        current = head
        while current is not None:
            next_node = current.next 
            current.next = previous
            previous = current #previous后移 
            current = next_node #current后移，previous和current必须并驾齐驱！！！
        return previous
```

#### [817. 链表组件](https://leetcode-cn.com/problems/linked-list-components/)Medium

https://www.bilibili.com/video/BV1cW411o7Eu?spm_id_from=333.337.search-card.all.click

head: 0 -> 1 -> 2 -> 3 -> 4

nums = [0 , 3, 1, 4]

基本思路：遍历head中每个节点，如果该节点存在于nums中就说明可能是一个子linklist，如果在nums中存在就用1表示，不存在用0表示，则head可以表示为1->1->0->1->1。我们只需要找到0来断开head即可

```python
class Solution:
    def numComponents(self, head: Optional[ListNode], nums: List[int]) -> int:
        ans = 0
        nums = set(nums) #用set方便查找，用in可以查找是否存在该元素
        cur = head

        while head:
            if head.val in nums:
                if head.next is None or head.next.val not in nums: # 两种情况，一种是head下一个节点为None就说明到头了，第二种是head下一个节点的值不存在nums中，就是0的情况
                    ans += 1
            head = head.next
        
        return ans
```



## 三. 回溯算法

在搜索尝试过程中寻找问题的解，当发现已不满足求解条件时，就“回溯”返回，尝试别的路径。回溯法是一种选优搜索法，按选优条件向前搜索，以达到目标。但当探索到某一步时，发现原先选择并不优或达不到目标，就退回一步重新选择，这种走不通就退回再走的技术为回溯法。

**递归与回溯相辅相成**

#### [1079. 活字印刷](https://leetcode-cn.com/problems/letter-tile-possibilities/)



#### [22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        def backtracking(self, result, n, left, right, str):
            if left < right: # 当右括号数量大于左括号时，直接return即可无需再往下走
                return

            if right == left == n: # 当左右括号数量等于n时说明构造完毕，把str添加到result中return即可
                result.append(str)
                return
            
            if left > right: 
                backtracking(self, result, n, left, right+1, str+')')

            if left < n:
                backtracking(self, result, n, left+1, right, str+'(')


        result = []
        backtracking(self, result, n, 0, 0, '')
        return result
```



## 四. 双指针

#### [剑指 Offer 57. 和为s的两个数字](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

定义两个指针i,j，分别指向nums的头和尾

如果两指针的值相加小于target，说明左指针的值小了，要右移左指针（数组为sorted）

反之亦然

```python
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int] 
        """
        i, j = 0, len(nums) - 1
        while i < j:
            s = nums[i] + nums[j]
            if s > target: j -= 1
            elif s < target: i += 1
            else: return nums[i], nums[j]
        return []
```



## 五. 巧用max和min函数

#### [2124. 检查是否所有 A 都在 B 之前](https://leetcode-cn.com/problems/check-if-all-as-appears-before-all-bs/)

字符串中 每个 `'a'` 都出现在 每个 `'b'` 之前 **可以等价为** 最后一个'a'的index一定在第一个出现的'b'之前

```python
class Solution(object):
    def checkString(self, s):
        """
        :type s: str
        :rtype: bool
        """
        last_a = -1
        first_b = len(s)

        for i in range(len(s)):
            if s[i] == 'a':
                last_a = max(i, last_a) # max函数可以用来迭代寻找最大值
            if s[i] == 'b':
                first_b = min(i, first_b) # min函数可以用来迭代寻找最小值
            
        return last_a < first_b # 最后判断条件可以当作return条件
        
```



## 六. 哈希表

- Python中的哈希表1 **hashmap**:

  - 使用普通的字典时，用法一般是dict={},添加元素的只需要dict[element] =value即，调用的时候也是如此，dict[element] = xxx,但前提是element字典里，如果不在字典里就会报错。这时**defaultdict**就能排上用场了，defaultdict的作用是在于，当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值

    - ```python
      dict={}
      dict['age'] = 18
      print(dict['name']) -> 此时会发生KeyError，因为dict中没有name这个key
      ```

  - `dict =defaultdict(factory_function)` 这个factory_function可以是list、set、str等等，作用是当key不存在时，返回的是工厂函数的默认值，比如list对应[ ]，str对应的是空字符串，set对应set( )，int对应0，如下举例：

    - ```python
      from collections import defaultdict
       
      test_dict1 = defaultdict(int)
      test_dict2 = defaultdict(str)
      test_dict3 = defaultdict(set)
      test_dict4 = defaultdict(list)
      print(test_dict1[1])
      print(test_dict2[1])
      print(test_dict3[1])
      print(test_dict4[1])
      --------
      输出：
      0
       
      set()
      []
      ```
    
  - defaultdict()用法：

    - ```python
      s = [('red', 1), ('blue', 2), ('red', 3), ('blue', 4), ('red', 1), ('blue', 4)]
      d = defaultdict(set)
      for k, v in s:
          d[k].add(v)
      ```

    - ```python
      d = defaultdict(set)
      for i in d.values():
      		xxxxx
      ```

- Python中的哈希表2 **hashset**:

  - 集合set(): set是一个无序**不重复**的序列，会自动去掉重复的元素 / 不支持slice和index访问
  
    - ```python
      应用：去除list中重复元素
      >>> a = [11,22,33,44,11,22]
      >>> b = set(a)
      >>> b
      set([33, 11, 44, 22])
      ```
  
    - ```python
      基本用法：
      
      t.add('x')            # 添加一项
      s.update([10,37,42])  # 在s中添加多项
      
      t.remove('H') # 删除某项
      s.clear() # 清空s
      set4.pop() # 随机删除一个值
      
      a = t | s          # t 和 s的并集
      b = t & s          # t 和 s的交集
      c = t – s          # 求差集（项在t中，但不在s中）
      d = t ^ s          # 对称差集（项在t或s中，但不会同时出现在二者中）
      ```
    
  - {}: 可以使用大括号 **{ }** 或者 **set()** 函数创建集合，注意：创建一个空集合必须用 **set()** 而不是 **{ }**，因为 **{ }** 是用来创建一个空字典。
  
    - ```python
      >>> basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
      >>> print(basket)                      # 这里演示的是去重功能
      {'orange', 'banana', 'pear', 'apple'}
      >>> 'orange' in basket                 # 快速判断元素是否在集合内
      True
      ```
  
- Python中的哈希表3:

  - Counter类
  
  - ```python
    from collections import Counter
    
    a = Counter('gallahad')                		 	# 传进字符串
    b = Counter({'red': 4, 'blue': 2})      		# 传进字典
    c = Counter(cats=4, dogs=8)             		# 传进元组
    d = Counter(['a', 'b', 'c', 'b', 'a', 'b']) # 传进字符串
    print(list(a.elements()))
    print(list(b.elements()))
    print(list(c.elements()))
    print(list(d.elements()))
    print('--'*10)
    for key, value in a.items():
      print(key, value)
    print('--'*10)
    print(dict(a))
    print(dict(b))
    print(dict(c))
    print('--'*10)
    print(a.most_common(3))
    print(b.most_common(1))
    print(c.most_common(1))
    ```
  
  - ```shell
    ['g', 'a', 'a', 'a', 'l', 'l', 'h', 'd']
    ['red', 'red', 'red', 'red', 'blue', 'blue']
    ['cats', 'cats', 'cats', 'cats', 'dogs', 'dogs', 'dogs', 'dogs', 'dogs', 'dogs', 'dogs', 'dogs']
    ['a', 'a', 'b', 'b', 'b', 'c']
    --------------------
    g 1
    a 3
    l 2
    h 1
    d 1
    --------------------
    {'g': 1, 'a': 3, 'l': 2, 'h': 1, 'd': 1}
    {'red': 4, 'blue': 2}
    {'cats': 4, 'dogs': 8}
    --------------------
    [('a', 3), ('l', 2), ('g', 1)]
    [('red', 4)]
    [('dogs', 8)]
    ```
  
- C++中的哈希表1

  - unordered_set
  - 可以通过unordered_set.count(x)是否返回1来判断是否存在元素x

- C++中的哈希表1

  - Unordered_map


#### [01.02. 判定是否互为字符重排](https://leetcode-cn.com/problems/check-permutation-lcci/)

用一个哈希表即可

```python
class Solution:
    def CheckPermutation(self, s1: str, s2: str) -> bool:
        if len(s1) != len(s2):
            return False

        hashset = defaultdict(int)
        for a in s1:
            hashset[a] += 1

        for b in s2:
            hashset[b] -= 1

        for i in hashset.values():
            if i != 0:
                return False

        return True
```

#### [884. 两句话中的不常见单词](https://leetcode-cn.com/problems/uncommon-words-from-two-sentences/)

python的“另类”哈希表：用Counter当作哈希表

```python
class Solution:
    def uncommonFromSentences(self, s1: str, s2: str) -> List[str]:
        freq = Counter(s1.split()) + Counter(s2.split())
        print(freq) # Counter({'this': 2, 'apple': 2, 'is': 2, 'sweet': 1, 'sour': 1})
        
        ans = list()
        for word, occ in freq.items():
            if occ == 1:
                ans.append(word)
        
        return ans
```

#### [599. 两个列表的最小索引总和](https://leetcode-cn.com/problems/minimum-index-sum-of-two-lists/)

```python
class Solution(object):
    def findRestaurant(self, list1, list2):
        #value为列表的字典
        m = defaultdict(list)
        #把每个字符串对应的索引入字典
        for i in range(len(list1)):
            m[list1[i]].append(i)
        #把每个字符串对应的索引入字典
        for i in range(len(list2)):
            m[list2[i]].append(i)
        min_index = float("inf")
        result = []
        #遍历求两个value[0] + value[1]的和最小的str
        for key, value in m.items():
            if len(value) == 2:
                if value[0] + value[1] < min_index:
                    result = []
                    min_index = value[0] + value[1]
                    result.append(key)
                elif value[0] + value[1] == min_index:
                    result.append(key)
        return result
```

注意：把list当作类型传入defaultdict时，要用append来添加。不能通过index来访问元素

此时的字典为`{'Shogun': [0, 1], 'Tapioca Express': [1], 'Burger King': [2, 2], 'KFC': [3, 0]})`

value为list类型，如果用int类型当作字典查询时需要两个for

#### [355. 设计推特](https://leetcode-cn.com/problems/design-twitter/)

```python
from collections import defaultdict

class Twitter:

    def __init__(self):
        self.tweets = defaultdict(list)
        self.following = defaultdict(list)
        self.timestamp = 0

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.tweets[userId].append((tweetId, self.timestamp))
        if len(self.tweets[userId]) > 10:
            self.tweets[userId].pop(0)
        self.timestamp += 1

    def getNewsFeed(self, userId: int) -> List[int]:
        posts = self.tweets[userId][:]
        for following in self.following[userId]: # 遍历自己关注列表发过的twitter
            posts.extend(self.tweets[following]) # 把关注列表发过的twitter添加到list中
        posts.sort(key = lambda x : x[1])
        return [p[0] for p in posts[-10:]][::-1]

    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId != followeeId and followeeId not in self.following[followerId]:
            self.following[followerId].append(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followeeId in self.following[followerId]:
            self.following[followerId].remove(followeeId)



# Your Twitter object will be instantiated and called as such:
# obj = Twitter()
# obj.postTweet(userId,tweetId)
# param_2 = obj.getNewsFeed(userId)
# obj.follow(followerId,followeeId)
# obj.unfollow(followerId,followeeId)
```



#### [2133. 检查是否每一行每一列都包含全部整数](https://leetcode-cn.com/problems/check-if-every-row-and-column-contains-all-numbers/)

哈希表检查每一行和每一列

核心思想：如果某一行或列没有包换全部整数，就必然会有重复元素。也就是说，可以通过检查某元素是否存在于哈希表中，来判断是否每一行或列包含全部整数

```python
class Solution(object):
    def checkValid(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: bool
        """
        n = len(matrix)
        hashset = set()

        for i in range(n):
            hashset.clear()
            for j in range(n):
                if matrix[i][j] in hashset:
                    return False
                else:
                    hashset.add(matrix[i][j])
        for i in range(n):
            hashset.clear()
            for j in range(n):
                if matrix[j][i] in hashset:
                    return False
                else:
                    hashset.add(matrix[j][i])

        return True
```

#### [705. 设计哈希集合](https://leetcode-cn.com/problems/design-hashset/)

我们开辟一个大小为bucket 的数组，数组的每个位置是一个链表。当计算出哈希值之后，就插入到对应位置的链表当中。

注意几点

- 在链表尾部添加（addLast()）需要从头遍历，时间复杂度为O(n)，**在链表头部添加（addFirst()），时间复杂度为O(1)**

```python
class Node:

    def __init__(self, value, nextNode=None):
        self.val = value
        self.next = nextNode

class Bucket:

    def __init__(self):
        self.head = Node(-1)

    def exists(self, val):
        cur = self.head.next
        while cur:
            if cur.val == val:
                return True
            cur = cur.next
        return False

    def insert(self, val):
        if not self.exists(val):
            node = Node(val, self.head.next) # 插入头部，插入头部复杂度O1
            self.head.next = node # 比如-1(head)->2->5->1要插入3到头部时，head.next是2->5->1,先把3这个节点初始化val=3,next=2->5->1, 然后再把head.next指向3这个节点即可，就变成了-1(head)->2->3->5->1

    def delete(self, val):
        pre = self.head
        cur = self.head.next
        while cur:
            if cur.val == val:
                pre.next = cur.next
                return
            pre = cur
            cur = cur.next


class MyHashSet:

    def __init__(self):
        """ Initialize your data structure here. """
        self.m = 1009  # 1000以上第一个质数，桶数一般取质数，质数个的分桶能让数据更加分散到各个桶中
        self.bucket = [Bucket() for _ in range(self.m)]

    def _hash(self, key):
        return key % self.m

    def add(self, key: int) -> None:
        self.bucket[self._hash(key)].insert(key)

    def remove(self, key: int) -> None:
        self.bucket[self._hash(key)].delete(key)

    def contains(self, key: int) -> bool:
        """ Returns true if this set contains the specified element """
        return self.bucket[self._hash(key)].exists(key)
```



## 七. 栈

#### [1544. 整理字符串](https://leetcode-cn.com/problems/make-the-string-great/)

- ```shell
  ord() 函数是 chr() 函数（对于 8 位的 ASCII 字符串）的配对函数，它以一个字符串（Unicode 字符）作为参数，返回对应的 ASCII 数值，或者 Unicode 数值
  
  ord("中")    # 20013
  chr(20013) # '中'
  ```

- 判断两字符可以用`abs(ord(a)-ord(b)) == 32`来判断是否为大小写

- `"".join(list)`返回list组成的string

  

把所有的字符依次压栈，碰到ascii差的绝对值等于'a'和'A'的ascii差的绝对值的，就消掉。

```python
class Solution:
    def makeGood(self, s: str) -> str:
        diff = ord('a') - ord('A')
        stk = []
        for c in s:
            if len(stk) != 0 and abs(ord(stk[-1])-ord(c)) == diff:
                stk.pop()
            else:
                stk.append(c)
        return "".join(stk)
```

#### [132 模式](https://leetcode-cn.com/problems/132-pattern/)

一个简单的思路是使用一层从左到右的循环固定 3，遍历的同时维护最小值，这个最小值就是 1（如果固定的 3 不等于 1 的话）。 接下来使用另外一个嵌套寻找枚举符合条件的 2即可。 这里的符合条件指的是大于1且小于 3。这种做法的时间复杂度为 O(n^2)，并不是一个好的做法，我们需要对其进行优化。132可表示为min, max, mid

实际上，我们也可以枚举 2 的位置，这样目标变为找到一个大于 2 的数和一个小于 2 的数。由于 2 在序列的右侧，因此我们需要从右往左进行遍历。又由于题目只需要找到一个 312 模式，因此我们应该贪心地选择尽可能大的 2（只要不大于 3 即可），这样才更容易找到1（换句话说不会错过 1）。

首先考虑找到 32 模式。我们可以使用从右往左遍历，遇到一个比后一位大的数。我们就找到了一个可行的 32 模式。

因此我们使用**单调栈**来解决这个问题：**单调栈是一个递减栈**，如果新加入的元素比top元素大，就记录下来当作max元素，top当mid，这样保证了找到max和mid(上升趋势)，继续加入元素直到找到一个比max, mid都小的元素就说明找到了132 pattern

https://www.bilibili.com/video/BV1SZ4y1x74J?spm_id_from=333.337.search-card.all.click

min, max, mid是波峰的样子，也就是说要找到一个突然增高的波峰，波峰左右是递减的才能实现，利用递减的单调栈，如果碰到比当前值大的数就做处理，碰到递减的就一直压栈即可。**所以单调栈可以解决突然有波峰的这种变化趋势**

```python
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        s2, stack = float("-inf"), []

        for i in range(len(nums) - 1, -1, -1):
            if nums[i] < s2:
                return True
            while stack and stack[-1] < nums[i]:
                s2 = stack.pop()
            stack.append(nums[i])
        return False
```

#### [496. 下一个更大元素 I](https://leetcode-cn.com/problems/next-greater-element-i/)

单调栈

空栈压入，非空栈且压入元素大于栈顶元素，则把压入元素与栈顶元素放到map中，并把压入元素压入栈中

```python
class Solution(object):
    def nextGreaterElement(self, nums1, nums2):
        dic, stack = {}, []

        for i in range(len(nums2)):
            while stack and stack[-1] < nums2[i]:
                dic[stack.pop()] = nums2[i]
            stack.append(nums2[i])

        return [dic.get(x, -1) for x in nums1] # 如果字典中没有key，则默认返回-1
```

#### [739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/)

维护一个单调递减的栈，即栈顶的温度最低（实际栈中存储index，用temperatures[stack[i]]获取温度)

如果当前温度temperatures[i]比栈顶高，则栈内所有比它低的元素都可以出栈（因为已经找到了下一个最高温度），栈内记录各自的index，所以可以就直接计算i-stack[-1]即可；
如果当前温度temperatures[i]比栈顶低，则直接入栈；
栈当前为空，则直接入栈；

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        length = len(temperatures)
        ans = [0] * length
        stack = []
        for i in range(length):
            while stack and temperatures[i] > temperatures[stack[-1]]:
                prev_index = stack.pop()
                ans[prev_index] = i - prev_index
            stack.append(i)
            print(stack)
        return ans
```



## 八. 分治Divide and Merge

#### [53. 最大子数组和](https://leetcode-cn.com/problems/maximum-subarray/)







## 九. 树

#### [897. 递增顺序搜索树](https://leetcode-cn.com/problems/increasing-order-search-tree/)

1. 先中序遍历，把结果放在数组中；
2. 然后修改数组中每个节点的左右指针：把节点的左指针设置为 null，把节点的右指针设置为数组的下一个节点。

下面的代码中，使用了 dummy （哑节点），它一般在链表题中出现。在链表题目中，我们为了防止链表的头结点发生变化之后，不好维护头结点，我们设置 dummy 从而保证头结点不变。这个题目中设置了 dummy ，从而保证了在新的树中，dummy 是根节点，最终返回的时候，要返回的是 dummy.right。

```python
class Solution(object):
    def increasingBST(self, root):
        self.res = []
        self.inOrder(root)
        if not self.res:
            return 
        dummy = TreeNode(-1)
        cur = dummy
        for node in self.res:
            node.left = node.right = None #储存在list里的元素，可能有左孩子或右孩子，在这一起清除了
            cur.right = node
            cur = cur.right
        return dummy.right
    
    def inOrder(self, root):
        if not root:
            return
        self.inOrder(root.left)
        self.res.append(root)
        self.inOrder(root.right)

```

##  十. 二分查找

- 如果用`mid=(left+right)/2`，在运行二分查找程序时可能溢出超时。因为如果left和right相加超过int表示的最大范围时就会溢出变为负数。所以如果想避免溢出，不能使用mid=(left+right)/2，应该使用`mid=left+(right-left)/2`

- **二分查找只适合 有序集合 使用，如果有重复元素可能返回多个值**

- **注意区间问题**

- `[left .....mid -1] mid [mid+1....right]`：因为right = numsSize-1

  ``````c++
  int binarySearch(int* nums, int numsSize, int target){
  	int left = 0, right = numsSize-1;
  	while(left <= right){ # 因为每次循环的搜索区间是[left,right]，区间内还有值(right是闭区间，可能还有right没有搜索),所以搜索不能停。除非left>right才表示区间内没有值可以停止搜索
  		int mid = left + (right - left)/2;
  		if(nums[mid] == target){
  			return mid;
  		}
  		else if(nums[mid] > target){
  			right = mid - 1;
  		}
  		else left = mid + 1;
  	}
  	return -1;
  }

- `[left....mid) mid [mid+1....right)`：因为right = numsSize，不包括最后一个值

  `````c++
  int binarySearch_2(int* nums, int numsSize, int target){
      int left = 0, right = numsSize;
      while(left < right){ #因为每次循环的搜索区间是[left,right), 终止的条件是 left == right，此时搜索区间 [left, left) 为空，所以可以正确终止
          int mid = left + (right - left)/2;
          if(nums[mid] == target){
              return mid;
          }
          else if(nums[mid] > target){
              right = mid;
          }
          else left = mid + 1;
      }
      return -1;
  }

- 怎么判断终止条件？
  - 循环直至区间左右端点相同时，如果`nums[right]`不含值时，就说明`nums[left]`和`nums[right]`都不包含值，就可以结束搜索。此时的条件是左闭右开，结束条件为：while (left < right)

#### [278. 第一个错误的版本](https://leetcode-cn.com/problems/first-bad-version/)

```c++
class Solution {
public:
    int firstBadVersion(int n) {
        int left = 1, right = n; // 注意这里是从1开始，因为题目中是从1...n
        while (left < right) { // 循环直至区间左右端点相同
            int mid = left + (right - left) / 2; // 防止计算时溢出
            if (isBadVersion(mid)) {
                right = mid; // 答案在区间 [left, mid] 中
            } else {
                left = mid + 1; // 答案在区间 [mid+1, right] 中
            }
        }
        // 此时有 left == right，区间缩为一个点，即为答案
        return left;
    }
};
```

#### [LCP 18. 早餐组合](https://leetcode-cn.com/problems/2vYnGI/)

假如题目里给定的 xx是100，我们从 staple 数组中选择的值为60，那么从 drinks 数组中只要选择的值不超过40即可，因此我们只需要找到这个临界值即可，找到了临界值，就已知了给定 staple 而选择的 drinks 的值有多少个。因此，我们先对 staple和 drinks进行排序，我们遍历 staple 中的每一个值，再找 drinks 中的临界值，由于我们已经对数组进行了排序，我们找临界值的方法可以从原先的遍历改为二分搜索，这样便可以节省很大一部分时间。

```python
class Solution:
    def breakfastNumber(self, staple: List[int], drinks: List[int], x: int) -> int:
        drinks.sort()
        staple.sort()

        res = 0
        for s in staple:
            l, r = 0, len(drinks)-1
            while l <= r:
                mid = l + (r - l) // 2
                if drinks[mid] + s <= x :
                    l = mid + 1
                else:
                    r = mid - 1
            res += l
        
        return res % 1000000007
```







## 十一. 排序算法

### 快速排序

**快速排序**是一种随机化算法，pivot的选取决定了快速排序的运行时间

1.从数组中选择一个元素作为pivot
2.重新排列数组，小于pivot的在pivot的左边，大于pivot的在其右边。
3.递归地对划分后的左右两部分重复上述步骤。

<img src="https://www.ranxiaolang.com/static/python_algorithm/images/%E5%BF%AB%E9%80%9F%E6%8E%92%E5%BA%8F.jpg" alt="快速排序– Dominc" style="zoom:50%;" />

```python
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:

        def partition(arr, low, high):
            pivot_idx = random.randint(low, high)                   # 随机选择pivot，随机选取pivot, 通常能得到比较好的结果
            arr[low], arr[pivot_idx] = arr[pivot_idx], arr[low]     # pivot放置到最左边，选取最左边第一个为pivot
            pivot = arr[low]                                        

            left, right = low, high     # 双指针
            while left < right:
                
                while left<right and arr[right] >= pivot:          # 找到右边第一个小于pivot的元素
                    right -= 1
                arr[left] = arr[right]                             # 并将其移动到left处
                
                while left<right and arr[left] <= pivot:           # 找到左边第一个大于pivot的元素
                    left += 1
                arr[right] = arr[left]                             # 并将其移动到right处
            
            arr[left] = pivot       # pivot放置到中间left=right处
            return left
        

        def quickSort(arr, low, high):
            if low >= high:         # 递归结束，跳出的条件时low大于等于high
                return  
            mid = partition(arr, low, high)     # 以mid为分割点，右边元素>左边元素
            quickSort(arr, low, mid-1)          # 递归对mid两侧元素进行排序
            quickSort(arr, mid+1, high)
        

        quickSort(nums, 0, len(nums)-1)         # 调用快排函数对nums进行排序
        return nums
```

### 堆排序

1. 完全二叉树
2. parent > children

Heapify：从头节点开始，在parent和两个children节点中找到最大的那个节点与parent交换，继续在下面的子树中完成该操作即可

第i节点的parent为`(i-1)/2`, children为`c1 = 2i+1`,`c2 = 2i + 2` 

python中的heapq构建最小堆：**python内建的heapq只能构建最小堆，无法构建最大堆，如果想构建最大堆，取相反数即可**

```python
import heapq

minheap = []
heapq.heapify(minheap) #自动构建最小堆

heapq.headpush(minheap, 10)
heapq.headpush(minheap, 9)
heapq.headpush(minheap, 8)
heapq.headpush(minheap, 2)
heapq.headpush(minheap, 1)
heapq.headpush(minheap, 11)

print(minheap) # [1,2,9,10,8,11]

heapq.headpop(minhead) # pop出最小的堆顶元素

while len(minhead) != 0:
  print(heapq.headpop(minhead))
  
heapq.nlargest(n, iterable, key=None)
#返回一个列表, 为根据key作为筛选条件从可迭代对象iterable中筛选的最大的n个元素
 
heapq.nsmallest(n, iterable, key=None)
#返回一个列表, 为根据key作为筛选条件从可迭代对象iterable中筛选的最小的n个元素
```

python手动实现堆排序:

1. 先从最后一个parent节点(n//2)-1通过headpify构建最大堆
2. 交换头节点与最后一个节点
3. 重新headpify头节点

```python
# 从i节点开始heapify，n为heap的size
def heapify(arr, n, i):
	largest = i # Initialize largest as root
	l = 2 * i + 1	 # left = 2*i + 1
	r = 2 * i + 2	 # right = 2*i + 2
	
  # 从左右孩子和parent中找到最大的数
	if l < n and arr[i] < arr[l]:
		largest = l
	if r < n and arr[largest] < arr[r]:
		largest = r

	# 如果parent不是最大的，就要跟孩子交换位置
	if largest != i:
		arr[i],arr[largest] = arr[largest],arr[i] # swap

		# Heapify the root.
		heapify(arr, n, largest)

# The main function to sort an array of given size
def heapSort(arr):
	n = len(arr)
	# Build a maxheap，Since last parent will be at ((n//2)-1) we can start at that location.
	for i in range(n // 2 - 1, -1, -1):
		heapify(arr, n, i)
	# swap -> remove -> heapify
	for i in range(n-1, 0, -1):
		arr[i], arr[0] = arr[0], arr[i] # swap
		heapify(arr, i, 0) # i是不断减少的，就是说用i来表示remove的过程，只需要heapify除去最后一个已经交换过的node剩余所有的node,相当于remove掉了最后一个node

# Driver code to test above
arr = [12, 11, 13, 5, 6, 7]
heapSort(arr)
n = len(arr)
print ("Sorted array is")
for i in range(n):
	print ("%d" %arr[i]),
```

![procedures for implementing heap sort](https://cdn.programiz.com/cdn/farfuture/VicaT2DyDXxbtM88OYklajepD4hkdSumEHTg2nBwe7s/mtime:1586942728/sites/tutorial2program/files/heap_sort.png)

python手写大顶堆：

```python

class Maxheap(object):
    def __init__(self, maxsize):
        self.maxsize = maxsize  # 堆的大小
        self._elements = [0] * maxsize  # 初始化堆
        self._count = 0  # 索引

    def add(self, value):
        """
        往堆里加新的值，相当于在list末尾添加值，然后siftup维护大根堆从上到下从大到小
        :param value:
        :return:
        """
        if self._count >= self.maxsize:
            raise Exception('full')
        self._elements[self._count] = value  # 放到末尾
        self._count += 1  # 索引加一，这是当前的下一个索引
        self._siftup(self._count - 1)  # siftup将当前索引值维护到堆的位置

    def extract(self):
        """
        提取堆顶元素，相当于heapq.heappop(heap),提取0位置元素然后用list末尾的元素补到0位置上，siftdown维护大根堆
        :return:
        """
        if self._count <= 0:
            raise Exception('empty')
        value = self._elements[0]  # 记录堆顶值
        self._count -= 1
        self._elements[0] = self._elements[self._count]  # 末尾移到堆顶
        self._siftdown(0)  # 从上到下维护堆
        return value

    def _siftup(self, index):
        if index > 0:
            parent = (index - 1) >> 2  # 当前索引的父索引
            if self._elements[index] > self._elements[parent]:  # 当前值大于父，需要替换
                self._elements[index], self._elements[parent] = self._elements[parent], self._elements[index]
                self._siftup(parent)  # 加入的值换到了父索引位置，继续向上看是不是比上一层的父更大

    def _siftdown(self, index):
        left = index << 2 | 1  # 左子树索引
        right = index << 2 | 2  # 右子树索引
        new_index = index  # 用一个新索引，后面观察需不需要换
        if right < self._count:  # 有左右子树的情况
            if self._elements[left] <= self._elements[index] and self._elements[right] <= self._elements[index]:  # 当前比左右都大，不用操作
                pass
            else:
                if self._elements[left] >= self._elements[right]:
                    new_index = left  # 左边更大，且左边大于当前，准备用左边跟当前索引换
                else:
                    new_index = right
        elif left < self._count:  # 只有左子树
            if self._elements[left] >= self._elements[index]:
                new_index = left
        if new_index != index:  # 需要换
            self._elements[new_index], self._elements[index] = self._elements[index], self._elements[new_index]
            self._siftdown(new_index)


if __name__ == '__main__':
    import random

    seq = list(range(10))
    random.shuffle(seq)
    print(seq)
    heap = Maxheap(len(seq))
    for i in seq:
        heap.add(i)

    res = []
    for i in range(10):
        res.append(heap.extract())
    print(res)
    #[9, 7, 3, 0, 8, 4, 1, 5, 2, 6]
		#[9, 7, 6, 5, 4, 3, 8, 2, 1, 0]
```



#### [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

对于 topk 问题：**最大堆求topk小，最小堆求 topk 大。**

取topK的元素常见套路：构造大小为 k 的小顶堆

https://www.youtube.com/watch?v=vIXf2M37e0k

```python
def findKthLargest(nums, k):
    heap = [x for x in nums[:k]]
    heapq.heapify(heap)  # 先取前k个元素构成len等于k的最小堆

    n = len(nums)
    for i in range(k, n): # 遍历k到n的元素，如果该元素比堆顶大，就把堆顶pop，把该元素push进head中
        if nums[i] > heap[0]:
            heapq.heappop(heap)  # 删除堆顶最小元素
            heapq.heappush(heap, nums[i])
    return heap[0]  # 返回堆顶即第k个最大元素
```

#### [347. 前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)

对于 topk 问题：**最大堆求topk小，最小堆求 topk 大。**

先用哈希表存储出现频率，然后维护一个k大小的小顶堆

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:

        map_ = collections.Counter()

        for i in nums:
            map_[i] += 1
        
        #对频率排序
        #定义一个小顶堆，大小为k
        pri_que = [] #小顶堆
        
        #用固定大小为k的小顶堆，扫面所有频率的数值
        for key, freq in map_.items():
            heapq.heappush(pri_que, (freq, key)) #heapq模块可以接受元组对象，默认元组的第一个元素作为priority，即按照元组的第一个元素构成小根堆
            if len(pri_que) > k: #如果堆的大小大于了K，则队列弹出，保证堆的大小一直为k
                heapq.heappop(pri_que)
        
        #找出前K个高频元素，因为小顶堆先弹出的是最小的，所以倒叙来输出到数组
        result = [0] * k
        for i in range(k-1, -1, -1):
            result[i] = heapq.heappop(pri_que)[1] #[1]是取出tuple的第二个值，第二个值才是频率(3,1)
        return result
```

heapq模块可以接受元组对象，默认元组的第一个元素作为`priority`，即按照元组的`第一个`元素构成小根堆

如果想要按照元组中的`其他元素`构成小根堆，在原来基础上加个`优先级`即可：

```
heappush(Q, (priority,tuple))
priority = key(tuple[i])
key=lambda x:f(x)
```

#### [451. 根据字符出现频率排序](https://leetcode-cn.com/problems/sort-characters-by-frequency/)

构建最大堆，记得取负号

```python
def frequencySort(s):
    """
    :type s: str
    :rtype: str
    """
    count = collections.Counter(s)
    items = [(-val, key) for key, val in count.items()] # items: [(-1, 't'), (-1, 'r'), (-2, 'e')]
    heapq.heapify(items)
    res = ""
    while items:
        val, key = heapq.heappop(items)
        res += key * (-val) # 乘以频率
    return res
```



### 桶排序

#### [451. 根据字符出现频率排序](https://leetcode-cn.com/problems/sort-characters-by-frequency/)

按一定的属性进行排序，可以想到桶排序

- 用 哈希表 统计字符串中各个字符出现的次数，保留最大次数 maxCount，用于建桶
- 建桶，用数组，数组最大长度为 maxCount，可以保证 桶 可以囊括所有字母
- 把元素按照出现的次数放入对应的桶 eg: bucket[2] = {a, b} 指 a，b 出现的次数为 2 , 放入对应的桶中
- 把桶按倒序倒出

```python
def frequencySort(s):
    """
    :type s: str
    :rtype: str
    """
    # 构建字典统计频数
    count_dict = collections.Counter(s)
    # 创建桶数组
    bucket = [[] for _ in range(len(s) + 1)] # bucket: [[], [], [], [], []]
    # 按频数大小分别放进各个桶里,桶的索引(下标)为字符出现的频数,桶里的值为该字符
    for key, value in count_dict.items():
        bucket[value].append(key * value) # bucket: [[], ['t', 'r'], ['ee'], [], []]
    # 逆序读取桶里的字符，即按照频数大小降序排列读取
    res = []
    for index in range(len(bucket) - 1, -1, -1):
        if bucket[index]:
            res.extend(bucket[index])
    return ''.join(res)
```

```cpp
class Solution {
public:
    string frequencySort(string s) {
        int size = s.size();
        if (size == 0 || size == 1) return s;
        // 统计出现的最大频率，来建立桶
        unordered_map<char, int> hash_map;
        int maxCount = INT_MIN;
        for (int i = 0; i < size; i++) {
            char cur = s[i];
            hash_map[cur]++;
            maxCount = max(maxCount, hash_map[cur]);
        }
        // 建立桶
        vector<vector<char>> bucket(maxCount + 1);
        // 把元素按照出现的次数放入对应的桶 eg bucket[2] = {a, b} 指 a，b 出现的次数为 2
        for (auto element : hash_map) {
            char curChar = element.first;
            int count = element.second;
            for (int i = 0; i < count; i++) {
                bucket[count].push_back(curChar);
            }
        }
        // 把元素按倒序倒出
        string res = "";
        for (int i = maxCount; i >= 0; i--) {
            for (int j = 0; j < bucket[i].size(); j++) {
                res += bucket[i][j];
            }
        }
        return res;
    }
};
```

## 十二. 贪心

#### [452. 用最少数量的箭引爆气球](https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/)

按照结束坐标**排序**，就可以保证气球位置有序排列

```python
class Solution(object):
    def findMinArrowShots(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """

        if points is None:
            return 0

        points.sort(key = lambda x: x[1]) # 已经排序过，就可以用贪心来做!!!
        pos = points[0][1]
        ans = 1
        for ballon in points:
            if ballon[0] > pos:
                pos = ballon[1]
                ans += 1

        return ans
```



## 十三. 记忆化搜索

记忆化搜索就是增加一个数组空间N来记录值。通过一定的空间复杂度增加，减少时间复杂度，空间换时间



## 十四. 动态数组

#### [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

https://www.bilibili.com/video/BV1tz4y1d7XM?spm_id_from=333.337.search-card.all.click

从0到amount每个都是当前使用最少硬币，所以amount自然也是最少硬币组成

比如amount=5，coins=[1，2，5]。5由x+1，y+2，z+5组成，x=4，y=3，z=0，再往下找x，y和z的最小组成即可

递推关系为：`dp[j] = min(dp[j], dp[j - coin] + 1) ` dp[j - coin]表示x，y和z

```python
import math
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # 初始化
        dp = [amount + 1]*(amount + 1)
        dp[0] = 0
        # 遍历物品
        for j in range(1, amount + 1): #遍历每一种可能的amount
            # 遍历背包
            for coin in coins: #遍历每一种组成的coin
                if j >= coin: #要确保j大于等于coin才能相减
                	dp[j] = min(dp[j], dp[j - coin] + 1)
        return dp[amount] if dp[amount] < amount + 1 else -1

```

#### [518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/)

https://www.bilibili.com/video/BV1kX4y1P7M3?spm_id_from=333.337.search-card.all.click

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0] *(amount+1)
        dp[0] = 1

        for coin in coins:
            for j in range(coin, amount+1):
                dp[j] += dp[j - coin]
        
        return dp[amount]
```

## Python Tricks

- [1640. 能否连接形成数组](https://leetcode-cn.com/problems/check-array-formation-through-concatenation/)：两个list顺序和元素都相同才会返回True



## C++ Tricks

