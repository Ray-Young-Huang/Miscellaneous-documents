```
TestProj                       # Qt 项目根目录
├─ TestProj.pro                # Qt 项目配置文件(包含编译信息、依赖库等)
├─ 头文件                      # 头文件目录
│  └─ mainwindow.h             # 主窗口头文件(类声明、成员变量和函数声明)
├─ 源文件                      # 源代码目录
│  ├─ main.cpp                 # 程序入口文件(包含 main 函数)
│  └─ mainwindow.cpp           
│     # 主窗口实现文件(类成员函数的具体实现)
└─ 界面文件                    # UI 设计文件目录
   └─ mainwindow.ui            
      # 主窗口界面设计文件(可视化设计的 UI 布局)
```

```cpp
// mainwindow.h 声明槽函数
private slots:
    void on_pushButton_new_clicked();
```

```cpp
// mainwindow.cpp 构造函数连接信号和槽函数
connect(ui->pushButton_load, &QPushButton::clicked, this, &MainWindow::on_pushButton_clicked);
```

```cpp
// mainwindow.cpp 实现槽函数
void MainWindow::on_pushButton_new_clicked()
{
    // 控制台输出信息
    qDebug("Hello World!");
}
```

```cpp
// 文件加载相关的头文件
#include <QFileDialog>   // 用于打开文件对话框
#include <QFile>         // 用于文件操作
#include <QTextStream>   // 用于文本流读取
#include <QMessageBox>   // 用于显示错误消息
```

```cpp
// 加载文件对话框
QString fileName = QFileDialog::getOpenFileName(
   this,                           // 父窗口
   "打开文本文件",                  // 对话框标题
   "",                             // 默认打开路径(空表示上次路径)
   "文本文件 (*.txt);;所有文件 (*.*)"  // 文件过滤器
);
```
```cpp
// 创建文件对话框
QString fileName = QFileDialog::getSaveFileName(
   this,
   "新建文本文件",
   "",
   "文本文件 (*.txt);;所有文件 (*.*)"
);
```

```cpp
// 文件内容读取的标准流程
// 为文件创建QFile对象
QFile file(fileName);
// 打开文件
file.open(QIODevice::ReadOnly);
// 为文件创建QTextStream流对象
QTextStream in(&file);
// 读取所有内容
QString content = in.readAll();
// 读取完成后关闭文件
file.close();
```

```cpp
// 文件内容写入的标准流程
if (!file.open(QIODevice::WriteOnly)) {
   QMessageBox::warning(this, "错误", "无法保存文件：" + file.errorString());
   return;
}
// 创建文本流
QTextStream out(&file);
// 从textEdit获取文本内容并写入文件
out << ui->textEdit->toPlainText();
// 关闭文件
file.close();
```