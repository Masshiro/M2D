# Cardinality Moment Estimator: Project Manual 
> Chinese Edition

*Conan* is a portable package manager, intended for C and C++ developers, but it is able to manage builds from source, dependencies, and precompiled binaries for any language.

Conan是一个跨平台的包管理工具，面向c和c++开发者，但是他能够从源码，依赖和预编译二进制包。

1. conan是个 c++/c 的包管理工具，基于python开发，开源。
2. conan需要编写conanfile.txt来说明依赖。
3. conan search -r=conan指令可以在远程仓库搜索包
4. conan install 指令来根据conanfile.txt安装库文件
5. 最终生成文件：conanbuildinfo.txt
6. 编写cmake后编译工程
7. 完成使用

## 项目组织结构树状图
```shell
.
├── CMakeLists.txt
├── README.md
├── conanfile.txt
├── data
│   ├── 130input_list.txt
│   ├── 5input_list.txt
│   └── caida/
├── experiment
│   ├── CMakeLists.txt
│   ├── hierarchy
│   └── sketch
├── main.cpp
├── result
│   ├── hierarchy
│   └── sketch
├── src
│   ├── CMakeLists.txt
│   ├── filter
│   ├── hierarchy
│   ├── sketch
│   └── utils
└── test
    ├── CMakeLists.txt
    ├── hierarchy
    └── sketch
```

## Linux下的Python3安装
```shell
sudo apt update
sudo apt upgrade
sudo apt install python3
sudo apt install python3-pip
```

## Conan安装：

```shell
$ sudo python3 -m pip install conan

$ conan --version  // 看到提示信息后说明安装成功
```


## CMake安装：

安装 cmake 后，输入 cmake -version 查看版本信息，看到提示信息后说明安装成功：

```shell
$ sudo apt install cmake

$ cmake -version
```


## 构建项目：

### 配置`conan profile`文件

此处需要改一下 Conan 中的默认编译标准，我们使用 c++11 标准编译库文件，

```shell
$ cd ~/.conan/profiles

$ gedit default
```

若```.conan```目录下没有```profiles```文件夹，则需要自己新建一个名为```default```的profile

```shell
$ conan profile new default
```

将 libstdc++ 改为 libstdc++11，修改后的 default 文件内容为：

> ```
> [settings]
> os=Linux
> os_build=Linux
> arch=x86_64
> arch_build=x86_64
> compiler=gcc
> compiler.version=9.4
> compiler.libcxx=libstdc++11
> build_type=Release
> [options]
> [build_requires]
> [env]
> ```

### 安装依赖项
#### 1. 使用Terminal
- 需要在工程根目录下创建一个 build 目录:

```shell
$ mkdir build && cd build
```

- 安装缺失库
```shell
$ conan install .. --build missing
```

#### 2.使用CLion
此处无需在根目录下创建`build`目录，进入`cmake-build-debug`目录下即可：
```shell
$ cd cmake-build-debug/
```
随后使用相同的办法进行缺失库的安装：
```shell
$ conan install .. --build missing
```

### 编译
在`build`或在`cmake-build-debug`目录下
```shell
$ cmake ..  

$ make
```

## 运行项目
### 使用Terminal
由于配置原因，需要在工程根本目录下进行运行，再在`build/bin`文件夹中运行相关测试用例：
```shell
./build/bin/test_hyperloglog
./build/bin/on_vhll_bias_variance
```

### 使用CLion
可直接使用可视化界面进行构建以及运行.
- 实际同样可以使用CLion内置终端工具，运行`cmake-build-debug/bin/`文件夹中相应的可运行内容

### 查看结果
上述终端使用情形下必须在工程根目录下进行运行的原因在于，代码内部路径配置要求，若未遵守，则无法在`result`目录中查看。   
最终运行结果可以通过终端提示信息找到相应位置进行观察：
```shell
[2021-06-17 19:24:51.562] [info] task:12996 end
end work thread: 139696725370624
[2021-06-17 19:24:51.565] [info] task:13001 end
end work thread: 139696641443584
[2021-06-17 19:24:51.566] [info] task:12999 end
end work thread: 139696767334144
[2021-06-17 19:24:51.567] [info] ./result/on_vhll/r_err_vhll_4s_1024r_128c_6-17-19:24:51.txt
```

## WSL2 + CLion解决方案
- 默认已经安装好WSL2
### WSL2环境配置
该步骤根据上述内容进行设置，与Linux环境下配置无差异

### CLion配置
- 在CLion中，点击`Settings/Preferences | Build, Execuation, Deployment | Toolchains`
- 点击`+`，创建新的Toolchain并选择`WSL`
- 根据页面显示，可能需要在`WSL`中进行相关安装（如`gdb`）
- 当配置完整加载后，将`WSL`设为默认toolchain（拖拽至列表顶端）
- 切换`Terminal`为`wsl.exe`
    - `Settings | Tools | Terminal | Shell path`中更改


## 参考信息
### CLion使用Gitee
- 在插件市场中搜索并安装Gitee插件，其余设置同GitHub相似
- 建议使用Gitee的帐号登录，而非token方式
- 帐号无法使用手机号，使用邮箱即可

### CLion使用Conan
- 可参见conan文档
    - [CLion with Conan](https://docs.conan.io/en/latest/integrations/ide/clion.html#general-integration)

### CLion使用Git
- 参见CLion官方文档
    - [CLion with Git](https://www.jetbrains.com/help/clion/using-git-integration.html)

### Windows安装WSL2
- 第三方教程
    - https://segmentfault.com/a/1190000040143442
- 微软官方
    - https://docs.microsoft.com/en-us/windows/wsl/install

## Troubleshooting

### Issue 1: gcc版本问题
以上过程中如果出现

```
CMake Error at build/conanbuildinfo.cmake:959 (message):
  Detected a mismatch for the compiler version between your conan profile
  settings and CMake:

  Compiler version specified in your conan profile: 7

  Compiler version detected in CMake: 9.3

  Please check your conan profile settings (conan profile show
  [default|your_profile_name])

  P.S.  You may set CONAN_DISABLE_CHECK_COMPILER CMake variable in order to
  disable this check.
Call Stack (most recent call first):
  build/conanbuildinfo.cmake:1041 (conan_error_compiler_version)
  build/conanbuildinfo.cmake:1146 (check_compiler_version)
  build/conanbuildinfo.cmake:698 (conan_check_compiler)
  CMakeLists.txt:15 (conan_basic_setup)


-- Configuring incomplete, errors occurred!
```

#### Solution
这主要是由于conan profile中指定的编译器版本为gcc 7.0，而开发者实际运行环境中的gcc版本可能更高（并不影响正常使用），这时为了构建项目，则需要根据提示，在```conanbuildinfo.cmake```中添加一行：

```
set(CONAN_DISABLE_CHECK_COMPILER TRUE)
```

- 整个流程建议在 Linux 环境下执行，开发过程中自行对 CMakeLists.txt 和 conanfile.txt 进行调整改动即可。

### Issue 2： 无法找到包
在后续的使用中，出现了`gtest/1.8.1 unable found in remotes`的问题，也就是包在远程服务器中找不到。

#### Solution

针对`gtest/1.8.1`在`remote`找不到的问题，查询官方文档发现，Conan针对远程库有过一次更新，更换了新的ConanCenter，内容如下：

> **ConanCenter** (https://conan.io/center) is the main official repository for open source Conan packages. It is configured as the default **remote** in the Conan client, but if you want to add it manually:
>
> ```shell
> conan remote add conancenter https://center.conan.io
> ```
>
> **conan-center** was the official repository but is **no longer a default \**remote\**** in the Conan client and **its usage is completely discouraged**. This documentation is kept here only for reference purposes.
>
> ```shell
> conan-center: https://conan.bintray.com [Verify SSL: True]
> ```
>
> **It contains all the packages that were uploaded to ConanCenter before July 1st (new packages are no longer uploaded to this remote), as well as legacy packages with full reference (zlib/1.2.11@conan/stable). **
>
> 上述说明，第二个远程库从7月以后不再更新，已经弃用。

所以安装新版本的`Conan 1.42.0`会自动使用第一个链接，而7月前安装，当时使用的还是第二个链接。

因此解决方案非常简单，使用如下命令，将第二个远程库的链接加入进去：

```shell
conan remote add conan-center https://conan.bintray.com
```
再使用下面的指令更新一下刚添加的远程库链接即可：

```shell
conan remote update conan-center https://conan.bintray.com
```

### Issue 3： `conanfile.py`加载问题

```shell
ERROR: Error loading conanfile at '/home/sh/.conan/data/poco/1.9.4/_/_/export/conanfile.py': Unable to load conanfile in /home/sh/.conan/data/poco/1.9.4/_/_/export/conanfile.py
  File "/home/sh/.conan/data/poco/1.9.4/_/_/export/conanfile.py", line 97
    tools.get(**self.conan_data["sources"][self.version], 
    												^
SyntaxError: invalid syntax
```

根据Google查询得知，这个错误是因为Python2无法解析```self.conan_data```，需要使用Python3，回答者给出的建议是卸载python2所安装的conan，使用python3安装的conan，这一点有点坑，因为官方文档中，给出的命令是`python`，而`python`默认就是Python2。

```shell
python2 -m pip uninstall conan
python3 -m pip install -U conan
```
#### Solution

按照时间顺序，问题2应该是出现在问题1之前，注意一开始直接使用以下命令安装`conan`即可：

```shell
sudo python3 -m pip install conan
conan --version
```
