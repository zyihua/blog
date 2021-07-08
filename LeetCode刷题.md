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
public String convertToDec(String number) {
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
long long fastPower(long long base, long long power) {
    long long result = 1;
    while (power > 0) {
        if (power % 2 == 1) {
            result = result * base;
        }
        power = power / 2;
        base = (base * base);
    }
    return result;
}
~~~
快速幂算法的核心思想就是每一步都把指数分成两半，而相应的底数做平方运算
