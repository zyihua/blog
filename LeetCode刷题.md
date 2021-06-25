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
