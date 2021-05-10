# SVG图标

## Android studio使用svg图标

在res/drawable目录下右键，在菜单中选择New-Vector Asset;
在Asset Type中选择Local file，在path中选择本地资源，size默认选择24x24，Opacity选100%，Enable auto mirroring for RTL Layout不勾选；
点击next，finish后会在res/drawable目录下生成xml文件，用法同普通drawable资源相同。

## pathData属性

> 图标的绘制路径信息

| 命令 | 作用 | 举例 |
| ------ | ------ | ------ |
| M | move to 移动绘制点，作用相当于把画笔落在哪一点 | M66,66 |
| L | line to 直线 | L66,66 |
| Z | close 闭合，连接起点与终点 | Z |
| C | cubic bezier 三次贝塞尔曲线 | C66,66 66,66 66,66 |
| Q | quatratic bezier 二次贝塞尔曲线 | Q66,66 66,66 |
| A | ellipse 圆弧 |  |

命令区分大小写，大写代表后面的参数是绝对坐标，小写表示相对坐标，相对于上一个点的位置。参数之间用空格或逗号隔开。

## fillColor属性

> 图标的填充颜色

## strokeColor属性

> 曲线的颜色

## fillType属性

> 图标的填充策略

fillType=nonZero
从要填充的区域，任选一点，做射线；
绘制svg图形的曲线，顺时针穿过射线则加一，逆时针穿过射线则减一；
若最后的值不为0，则进行填充。

fillType=evenodd
从要填充的区域，任选一点，做射线；
绘制svg图形的曲线，穿过射线则加一；
若最后的值为奇数，则进行填充。

https://www.jianshu.com/p/89efdbe01ac9
