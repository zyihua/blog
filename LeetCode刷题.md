## 401. 二进制手表
Integer.bitCount(int i); // 计算i以二进制数表示时1的数量
### 算法原型
~~~
public static int bitCount(int i) {
    i = (i & 0x55555555) + ((i >>> 1) & 0x55555555);  // 计算每2位的1的数量
    i = (i & 0x33333333) + ((i >>> 2) & 0x33333333);  // 计算每4位的1的数量
    i = (i & 0x0f0f0f0f) + ((i >>> 4) & 0x0f0f0f0f);  // 计算每16位的1的数量
    i = (i & 0x00ff00ff) + ((i >>> 8) & 0x00ff00ff);  // 计算每32位的1的数量
    i = (i & 0x0000ffff) + ((i >>> 16) & 0x0000ffff);  // 计算每64位的1的数量
    return i;
}
~~~
先计算每2位的1的数量，0x55555555即‭0b01010101010101010101010101010101  
(i & 0x55555555)是每2位中后1位的1的数量，((i >>> 1) & 0x55555555)是每2位中前一位的1的数量  
### 优化后的源码
~~~
public static int bitCount(int i) {
    i = i - ((i >>> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >>> 2) & 0x33333333);
    i = (i + (i >>> 4)) & 0x0f0f0f0f;
    i = i + (i >>> 8);
    i = i + (i >>> 16);
    return i & 0x3f;
}
~~~
第一步：两个bit计算1的数量：0b11: 0b01 + 0b01 = 0b10 = 2, 0b10: 0b00 + 0b01 = 0b01 = 1。研究发现：2=0b11-0b1，1=0b10-0b1,可以减少一次位于计算：i = i - ((i >>> 1) & 0x55555555)  
第二步：暂时没有好的优化方法  
第三步：实际是计算每个byte中的1的数量，最多8（0b1000）个，占4bit，可以最后进行位与运算消位，减少一次&运算：i = (i + (i >>> 4)) & 0x0f0f0f0f  
第四,五步：同上理由，可以最后消位。但是由于int最多32（0b100000）个1，所以这两步可以不消位，最后一步把不需要的bit位抹除就可以了：i & 0x3f  

## 149. 直线上最多的点数
用辗转相除法或更相减损法求两个正整数的最大公约数（Greatest common divisor）
### 辗转相除法
~~~
public int gcd(int a, int b) {
    if (a % b == 0) {
        return b;
    } else {
        return gcd(b, a % b);
    }
}
~~~
原理：两个正整数a和b（a>b），它们的最大公约数等于a除以b的余数c和b之间的最大公约数。
递归计算余数c与较小数b之间的最大公约数，当余数c为0时，较小数b就是最大公约数。
### 更相减损法
~~~
public int gcd(int a, int b) {
    if (a == b) {
        return a;
    }
    if (a < b) {
        return gcd(a, b - a);
    } else {
        return gcd(b, a - b);
    }
}
~~~
原理：两个正整数a和b（a>b），它们的最大公约数等于a-b的差值c和较小数b的最大公约数。
递归计算差值c与较小数b的最大公约数，当差值c与较小数b相等时，最大公约数就是最终相等的两个数。
### 求最小公倍数
~~~
public int lcm(int a, int b) {
    return (a * b) / gcd(a,b);
}
~~~
两个数的乘积除以最大公约数得到的结果就是最小公倍数。

## 752. 打开转盘锁
广度优先搜索 (Breadth first search, BFS) 与深度优先搜索(Depth first search, DFS);  
A\*算法
### BFS
~~~
public static int BFS() {
    int step = 0;
    // 初始化队列
    Queue<String> queue = new LinkedList<String>();
    queue.offer("0000");
    // 初始化到过的状态的集合
    Set<String> seen = new HashSet<String>();
    seen.add("0000");

    while (!queue.isEmpty()) {
        ++step;
        int size = queue.size();
        // 这个for循环用来控制step的值
        for (int i = 0; i < size; ++i) {
            // 获取当前状态
            String status = queue.poll();
            // 遍历当前状态的下一状态
            for (String nextStatus : getNextStatus(status)) {
                // 去除到过的状态
                if (!seen.contains(nextStatus)) {
                    // 出口
                    if (nextStatus.equals(target)) {
                        return step;
                    }
                    // 将下一状态加入队列，并设置为到过
                    queue.offer(nextStatus);
                    seen.add(nextStatus);
                }
            }
        }
    }
    return -1;
}
~~~

### DFS
~~~

~~~

## 773. 滑动谜题
数组的深拷贝和浅拷贝
数组之间是否相等的比较
### 数组的深浅拷贝
~~~
// 深拷贝，用for循环遍历赋值，目标数组值改变不会影响原数组
// 浅拷贝，拷贝的数组值改变会影响原数组
Arrays.copyOf();
System.arraycopy();
数组.clone();
~~~
### 数组之间是否相等的比较
~~~
// 一维数组
Arrays.equals(数组a, 数组b);
// 多维数组
Arrays.deepEquals(数组a, 数组b);
~~~

## 168. Excel表列名称
十进制数转其他进制数，除留取余，逆序排列
其他进制数转十进制数，按权展开
### 十进制数转26进制数
~~~
public String convertToN(int number) {
    if (number == 0) {
        return "0";
    }
    StringBuilder res = new StringBuilder();
    while(number != 0) {
        res.append(number % 26);
        number  = number / 26;
    }
    return res.reverse().toString();
}
~~~
### 26进制数转十进制数
~~~
public long convertToDec(String number) {
    String numStr = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    StringBuilder string = new StringBuilder(number);
    char[] array = string.reverse().toString().toCharArray();
    long result = 0;
    for (int i = 0; i < array.length; i++) {
        int index = numStr.indexOf(array[i]);
        result += index * Math.pow(26, i);
    }
    return result;
}
~~~

## 37. 序列化二叉树
二叉树的广度优先遍历与深度优先遍历,遍历后的还原  
广度优先遍历又叫层次遍历  
深度优先遍历又分前序遍历，中序遍历，后序遍历  
### 二叉树的DFS
~~~
public void traverse(TreeNode root) {
    // 前序遍历
    traverse(root.left)
    // 中序遍历
    traverse(root.right)
    // 后序遍历
}
~~~
### 字串还原为二叉树
~~~
public TreeNode deserialize(String data) {
    String[] arr = data.split(",");
    Queue<String> queue = new LinkedList<String>();
    for(int i = 0; i < arr.length; i++){
        queue.offer(arr[i]);
    }
    return help(queue);
}
public TreeNode help(Queue<String> queue){
    String val = queue.poll();
    if(val.equals("null")){
        return null;
    }
    TreeNode root = new TreeNode(Integer.valueOf(val));
    root.left = help(queue);
    root.right = help(queue);
    return root;
}
~~~

## 726. 原子的数量
表达式的解析，通用的解法是使用递归或栈。本题中我们将使用栈解决。
### 栈
~~~
//核心代码
while (i < n) {
    char ch = formula.charAt(i);
    if (ch == '(') {
        i++;
        stack.push(new HashMap<String, Integer>()); // 将一个空的哈希表压入栈中，准备统计括号内的原子数量
    } else if (ch == ')') {
        i++;
        int num = parseNum(); // 括号右侧数字
        Map<String, Integer> popMap = stack.pop(); // 弹出括号内的原子数量
        Map<String, Integer> topMap = stack.peek();
        for (Map.Entry<String, Integer> entry : popMap.entrySet()) {
            String atom = entry.getKey();
            int v = entry.getValue();
            topMap.put(atom, topMap.getOrDefault(atom, 0) + v * num); // 将括号内的原子数量乘上 num，加到上一层的原子数量中
        }
    } else {
        String atom = parseAtom();
        int num = parseNum();
        Map<String, Integer> topMap = stack.peek();
        topMap.put(atom, topMap.getOrDefault(atom, 0) + num); // 统计原子数量
    }
}
//括号右侧字母或字符
public String parseAtom() {
    StringBuffer sb = new StringBuffer();
    sb.append(formula.charAt(i++)); // 扫描首字母
    while (i < n && Character.isLowerCase(formula.charAt(i))) {
        sb.append(formula.charAt(i++)); // 扫描首字母后的小写字母
    }
    return sb.toString();
}
//括号右侧数字
public int parseNum() {
    if (i == n || !Character.isDigit(formula.charAt(i))) {
        return 1; // 不是数字，视作 1
    }
    int num = 0;
    while (i < n && Character.isDigit(formula.charAt(i))) {
        num = num * 10 + formula.charAt(i++) - '0'; // 扫描数字
    }
    return num;
}
~~~
## 1418. 点菜展示表
HashMap的keySet()是乱序输出。
### set转array
~~~
String[] array = set.toArray(new String[0]);
~~~
对于返回值，如果参数指定的数组能够容纳 Set 集合的所有内容，就使用该数组保存 Set 集合中的所有对象，并返回该数组；否则，返回一个新的能够容纳 Set 集合中所有内容的数组。

## 1711. 大餐计数
用位运算判断一个数是否是2的幂，注意排除0
大数对质数1e9 + 7取余（也叫取模）
### 是否是2的幂
~~~
// 判断是否是2的幂
public static boolean isPowerOf2(int num) {
    if (num != 0 && (num & (num - 1)) == 0) {
        return true;
    }
    return false;
}
~~~
### 取模公式
~~~
(a+b) mod c
=(a mod c+ b mod c) mod c
=(a mod c+ b) mod c

(a*b*c)%d=(a%d*b%d*c%d)%d
~~~
### 排列组合之组合数公式
![image](https://user-images.githubusercontent.com/47979659/124879413-c930b880-dfff-11eb-87a7-3fd964aae318.png)

### 组合数取模
求(a / b) mod c  
(1)模运算
~~~
(a / b) mod c != (a mod c / b mod c) mod c
~~~
这里用到第一个概念“模运算”，模运算与基本四则运算有些相似，但是除法例外，所以引入第二个概念“逆元”。  
（2）逆元
~~~
(a / b) mod c = （a * (b的逆元)）mod c
~~~
如果(ax) mod c = 1， 那么x的最小正整数解就是a的逆元  
借助逆元我们可以把模运算的除法转为乘法，那如何求一个数的逆元呢，有两种方法：“拓展欧几里得算法”和“费马小定理”，这里介绍第三个概念“费马小定理”  
(3)费马小定理
~~~

~~~
![image](https://user-images.githubusercontent.com/47979659/124887593-e4072b00-e007-11eb-9af0-e3fb763a1728.png)  
因为题中c为质数，所以可以用费马小定理  
所以b的逆元就是power（b, p - 2）,如何求b的p-2次幂，这里引入第四个概念“快速幂”  
（4）快速幂
~~~
// java API计算一个数的幂
Math.pow(base, power);
// 快速幂（防止溢出一般做取模操作）
public static int fastPower(int base,  int power) {
    int result = 1;
    while (power > 0) {
        if (power % 2 == 1) {
            result = result * base % 100000000007;
        }
        power = power / 2;
        base = (base * base) % 100000000007;
    }
    return result;
}
~~~
快速幂算法的核心思想就是每一步都把指数分成两半，而相应的底数做平方运算

## 面试题 17.10. 主要元素
用摩尔投票法找出一个数组的主要元素（数组中占比超过一半的元素称之为主要元素）
~~~
public static int majorityElement(int[] nums) {
    // 摩尔投票法
    int candidate = -1;
    int count = 0;
    for (int num : nums) {
        if (count == 0) {
            candidate = num;
        }
        if (candidate == num) {
            count++;
        } else {
            count--;
        }
    }
    // 验证候选值是否是主要元素
    count = 0;
    int length = nums.length;
    for (int num : nums) {
        if (num == candidate) {
            count++;
        }
    }
    return count * 2 > length ? candidate : -1;
}
~~~
Boyer-Moore 投票算法的步骤如下：

维护一个候选主要元素 candidate 和候选主要元素的出现次数 count，初始时 candidate 为任意值，count=0；

遍历数组中的所有元素，遍历到元素 x 时，进行如下操作：

如果 count=0，则将x的值赋给 candidate，否则不更新 candidate 的值；

如果 x=candidate，则将 count 加 1，否则将 count 减 1。

遍历结束之后，如果数组 nums 中存在主要元素，则 candidate 即为主要元素。

## 1838. 最高频元素的频数
滑动窗口
### 滑动窗口
~~~
for (right < s.size()) {
    // 增大窗口
    window.add(s.right);
    right++;
    while(windows need shrink) {
        // 缩小窗口
        window.remove(s.left);
        left++;
    }
}
~~~

## 1893. 检查是否区域内所有整数都被覆盖
差分数组，适用于区间频繁修改，而且这个区间范围是比较大的，离线查询的情况
### 差分数组
~~~
// 返回数组nums的差分数组
public static int[] differentialArray(int[] nums) {
    int[] diff = new int[nums.length];
    diff[0] = nums[0];
    for (int i = 1; i < nums.length; i++) {
        diff[i] = nums[i] - nums[i - 1];
    }
    return diff;
}

// 返回差分数组diff的原数组
public static int[] restoresArray(int[] diff) {
    int[] res = new int[diff.length];
    res[0] = diff[0];
    for (int i = 1; i < diff.length; i++) {
        res[i] = res[i - 1] + diff[i];
    }
    return res;
}

// 修改[left, right]区间的值，都加3
public static int[] modifiesArray(int[] diff, int left, int right) {
    diff[left] += 3;
    diff[right + 1] -= 3;
    return diff;
}
~~~

## 1143. 最长公共子序列
动态规划法求最长公共子序列
### 最长公共子序列
~~~
int[][] dp = new int[m + 1][n + 1];
for (int i = 1; i <= m; i++) {
    char c1 = text1.charAt(i - 1);
    for (int j = 1; j <= n; j++) {
        char c2 = text2.charAt(j - 1);
        if (c1 == c2) {
            dp[i][j] = dp[i - 1][j - 1] + 1;
        } else {
            dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
        }
    }
}
return dp[m][n];
~~~

## 743. 网络延迟时间
迪杰斯特拉算法（Dijkstra）求最短路径
### Dijkstra
~~~
void dijkstra() {
    int INF = Integer.MAX_VALUE / 2;
    int[] dist = new int[N];
    boolean[] vis = new boolean[N];
    Arrays.fill(dist, INF);
    Arrays.fill(vis, false);
    dist[k] = 0;
    for (int p = 1; p <= n; p++) { // 一次循环确定一个点的最短路径
        int t = -1;
        for (int i = 1; i <= n; i++) { // 找出距离起点最近的点x
            if (!vis[i] && (t == -1 || dist[i] < dist[t]))
            t = i;
        }
        vis[t] = true; // 标记点x已经确定最短路径
        for (int i = 1; i <= n; i++) { // 通过点x更新其他所有点距离起点的最短距离
            dist[i] = Math.min(dist[i], dist[t] + w[t][i]);
        }
    }
}
~~~

## 551. 学生出勤记录 I
记录下奇妙的一行代码解法
### 一行代码
~~~
public boolean checkRecord(String s) {
    return s.indexOf("A") == s.lastIndexOf("A") && !s.contains("LLL");
}
~~~

## 704. 二分查找
有序数组，用二分查找更快速；二分查找仅限于有序数组。
二分查找思想很简单，但代码实现需要注意细节
二分查找分三种情况，比如{1,2,2,2,3}，当目标值为2时，是要查询哪个2呢。
此时普通的二分查找只能随缘返回下标了。
左侧边界的二分查找会返回最左侧的2的下标。
右侧边界的二分查找会返回最右侧的2的下标。
### 普通二分查找
~~~
public int search(int[] nums, int target) {
    int left = 0;
    int right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target){
            right = mid - 1;
        } else {
            return mid;
        }
    }
    return -1;
}
~~~
计算 mid 时需要防止溢出，代码中 left + (right - left) / 2 就和 (left + right) / 2 的结果相同，但是有效防止了 left 和 right 太大直接相加导致溢出

### 返回左侧边界的二分查找
~~~
public int searchLeft(int[] nums, int target) {
    int left = 0;
    int right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid - 1;
        } else if (nums[mid] == target) {
            // 别返回，锁定左侧边界
            right = mid - 1;
        }
    }
    // 最后要检查 left 越界的情况
    if (left >= nums.length || nums[left] != target)
        return -1;
    return left;
}
~~~

### 返回右侧边界的二分查找
~~~
public int searchRight(int[] nums, int target) {
    int left = 0;
    int right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid - 1;
        } else if (nums[mid] == target) {
            // 别返回，锁定右侧边界
            left = mid + 1;
        }
    }
    // 最后要检查 right 越界的情况
    if (right < 0 || nums[right] != target)
        return -1;
    return left;
}
~~~

## 600. 不含连续1的非负整数
01字典树
### dp + 01字典树求解
~~~
public int findIntegers(int n) {
    int[] dp = new int[31];
    dp[0] = dp[1] = 1;
    for (int i = 2; i < 31; ++i) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }

    int pre = 0, res = 0;
    for (int i = 29; i >= 0; --i) {
        int val = 1 << i;
        if ((n & val) != 0) {  // 节点为1时，即树高为i + 2的字典树有右子树时;
            n -= val;
            res += dp[i + 1];  // 加上树高为i + 2的字典树的左子树的路径数量（即根节点为0的满二叉树中，不包含连续1的从根节点到叶节点的路径数量）
            if (pre == 1) {   // 有连续1直接返回
                break;
            }
            pre = 1;
        } else {
            pre = 0;
        }

        if (i == 0) {
            ++res;
        }
    }

    return res;
}
~~~
状态转移方程：
dp[t]={ dp[t−1]+dp[t−2],t≥2
      { 1,              t<2
dp[t] 表示高度为 t+1、根结点为 0 的满二叉树中，不包含连续 1 的从根结点到叶结点的路径数量。
 
## 496. 下一个更大元素 I
放元素入栈，若找到比自己大的元素就出栈，否则继续入栈，直到找到比自己大的为止
### 单调栈
~~~
for (int i = 0; i < len2; i++) {
    while (!stack.isEmpty() && stack.peekLast() < nums2[i]) { // 找到比自己大的就出栈
        map.put(stack.removeLast(), nums2[i]);
    }
    stack.addLast(nums2[i]);
}
~~~

## 524. 通过删除字母匹配到字典里最长单词
双指针法求t是否是s的子序列
### 双指针法判断t是否是s的子序列
~~~
for (String t : dictionary) {
    int i = 0, j = 0;
    while (i < t.length() && j < s.length()) {
        if (t.charAt(i) == s.charAt(j)) {
            ++i;
        }
        ++j;
    }
    if (i == t.length()) {
        return t;
    }
}
return "";
~~~

## 212. 单词搜索 II
字典树
### 字典树
~~~
class Trie {
    private String val;
    private Trie[] childs = new Trie[26];
    Trie () {

    }

    public void insert(String word) {
        Trie cur = this;
        for (int i = 0; i < word.length(); i++) {
            int index = word.charAt(i) - 'a';
            if (cur.childs[index] == null) {
                cur.childs[index] = new Trie();
            }
            cur = cur.childs[index];
        }
        cur.val = word;
    }
}
~~~

## 292. Nim 游戏
桌子上有一堆石头。
你们轮流进行自己的回合，你作为先手。
每一回合，轮到的人拿掉 1 - 3 块石头。
拿掉最后一块石头的人就是获胜者。

如果堆里的石头数目为 4 的倍数时，你一定会输掉游戏
最优的选择是当前己方取完石头后，让剩余的石头的数目为 4 的倍数, 此时你只需要取走 x mod 4 个石头
### 巴什博奕
~~~
public boolean canWinNim(int n) {
    return n % 4 != 0;
}
~~~
