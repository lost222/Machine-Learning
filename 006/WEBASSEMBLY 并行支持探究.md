# WEBASSEMBLY 并行支持探究

## 摘要

WEBASEEMBLY是一种在浏览器里运行的， 类似汇编语言的中间码。 在2017年提出， 提出的目的是代替部分JavaScript在浏览器里的作用，提高前端的计算性能。 在2018年到2019年过程中逐渐支持了多线程和SIMD。 我想要测试性能如何。 

## 实验计划

1. WEBASSEMBLY已经有一个基于LLVM的编译器，能把C语言直接编译为WEBASSEMBLY， 已经支持在浏览器里多线程， 对多线程性能我的目标是：
   1. 最低标准：设计和实现类似前几次作业Pthtread和OpenMP的实验并不为难。 
   2. 中级目标：我现在还知道几个基于OpenCL的benchmark， 我希望能够解决编译的问题， 成功把这些东西在浏览器里跑起来， 测出比较权威的数据。
2. SIMD方面， 我现在已经搞明白从C的SIMD指令编译到WEBASSEMBLY的SIMD指令并不可行了。 我希望自己能做到这么几件事：
   1. 最低标准： 自己手写一些WEBASSEMBLY的SIMD代码， 获得一些数据。
   2. 中级目标：找到一些SIMD的C实现的benchmark， 或者找到一些适合SIMD使用的场景， 手写SIMD代码。
   3. 找到办法把C的SIMD代码编译为WEBASSEMBLY， 现在的状态是官方文档说支持， 但是实际上并不。 

## WEBASSEMBLY介绍

写这一节一方面是介绍一下WEBASSEMBLY， 另一方面也展示一下我在这个方向上已经做了一些探究， 主要是开始写开题报告的时间太晚，很多东西来不及写得很清楚。 

### WEBASSEMBLY诞生和支持

2017年诞生。现在已经被几大浏览器支持。

首次提出

Haas, A., Rossberg, A., Schuff, D. L., Titzer, B. L., Holman, M., Gohman, D., ... & Bastien, J. F. (2017, June). Bringing the web up to speed with WebAssembly. In *ACM SIGPLAN Notices* (Vol. 52, No. 6, pp. 185-200). ACM.

### WEBASSEMBLY的性能

大部分情况下能只比本地慢百分之五十， 比JS代码在浏览器里跑快一倍左右。

Jangda, A., Powers, B., Guha, A., & Berger, E. (2019). Mind the Gap: Analyzing the Performance of WebAssembly vs. Native Code. *arXiv preprint arXiv:1901.09056*.

### WEBASSEMBLY的应用前景

主要扣住它比JS快，做本来我们不想在前端里做的事情。 

比如给不愿意开发前段的小软件公司做网页端移植

Heil, S., Siegert, V., & Gaedke, M. (2018, June). ReWaMP: Rapid Web Migration Prototyping Leveraging WebAssembly. In *International Conference on Web Engineering* (pp. 84-92). Springer, Cham.



还有一个比较感兴趣的事情是， SIMD指令本身就是为了支持流媒体出现的， 结果长期以来的JS实际上是不支持SIMD的， 结果就是前端软件上对SIMD的支持其实不太好， WEBASSEMBLY能填补这个空白。

Park, J. T., Kim, H. G., & Moon, I. Y. (2018, June). Web Assembly Performance Analysis for Multimedia Processing in Web Environment. In *INTERNATIONAL CONFERENCE ON FUTURE INFORMATION & COMMUNICATION ENGINEERING* (Vol. 10, No. 1, pp. 288-290).