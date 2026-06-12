# 计算机视觉实验二：OpenCV 图像基础操作、调试（多文件版本）

## 实验环境
- 操作系统：Ubuntu / WSL2
- 开发工具：VS Code
- 编程语言：C++
- 依赖库：VS Code + Ubuntu (WSL) + C++ + OpenCV4

## 实验目的
1. 掌握 C++ 多文件项目结构的编写与编译方法。
2. 学会使用 VS Code 的 `tasks.json` 和 `launch.json` 配置编译与调试环境。
3. 实现 OpenCV 基础功能：图像读取、信息显示、灰度图转换、保存与显示。

## 项目结构
2023103402_孙勇超/
├── .vscode/               
│   ├── tasks.json          # 运行/调试任务
│   ├── launch.json         # 调试配置
├── build/        ← 编译输出
├── src/                    # 代码目录（多文件拆分）
│   ├── main.cpp      ← 主文件
│   ├── func.cpp      ← 功能文件（多文件）
│   └── func.h        ← 头文件
├── data/                   # 放你的图片
│   └── _20231211070702.jpg # 你的原图
├── output/                 # 生成的图片
└── README.md               

## 多文件编译/运行说明
本项目采用C++ 多文件模块化设计：
- `func.h`：声明函数
- `func.cpp`：实现图像信息输出、灰度转换功能
- `main.cpp`：主流程调用，完成读取、显示、保存

编译时必须将所有 `.cpp` 文件一起编译：
```bash
g++ -g main.cpp func.cpp -o build/app `pkg-config --cflags --libs opencv4`

## 学生信息
作者：孙勇超  
学号：2023103402