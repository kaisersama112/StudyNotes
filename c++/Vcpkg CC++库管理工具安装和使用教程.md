### 1、概述：

Vcpkg是可用于Windows，linux和MacOS上跨平台的 C/C++ 库管理工具，是一款开源工具。在此对vcpkg的安装做一个简要的说明，有关具体命令的使用，在安装成功后使用 vcpkg help 或者 vcpkg help[comment] 可以获取特定命令的命令帮助。更多内容请详见以下地址：

GitHub: https://github.com/microsoft/vcpkg

### 2、条件（Windows）：

前提条件：
1、 Windows7及以上系统
2、 Git
3、 Visual Studio 2015及以上版本
4、CMake 3.8.0及以上版本（如果没安装过，vcpkg会自动安装，一开始可忽略安装）

### 3、安装Visual Studio与Git：

假设你已经有安装Visual Studio，这里提示一点的是，如果使用的是 中文 界面的Visual Studio，还需要安装Visual Studio的 英文语言包 ，具体就是运行Visual Studio Installer——修改——语言包，安装英文语言包（很重要）

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020102516034432.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMDIzMjcw,size_16,color_FFFFFF,t_70#pic_center)


下载安装 Git ，安装好后将 git.exe 的路径添加到系统 path 环境变量当中去，这样就可以在Windows系统的 Win+R 的系统cmd命令行中使用vcpkg。我们主要使用的是 Git CMD ，当添加完Gti环境变量后，你也就可以使用Windos系统自带的cmd命令行工具。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201025173709822.png#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201025160707225.png#pic_center)

### 4、安装vcpkg：



以上只是准备工作，完成以上操作后正式进行vcpkg的安装过程。

首先你需要下载引导vcpkg，安装位置随意，但是为了之后方便与Visual Studio以及其他C/C++编译器链接，建议使用类似 C:\src 或者 C:\dev ，可以理解为在本地建立一个库的仓库，之后所有的安装都在该目录下，否则会由于路径的缘故会遇到某些端口构建系统的路径问题。

1、打开Git CMD命令行工具，使用 cd 到建立的目标路径（如：C:\src）；
2、输入命令：git clone https://github.com/Microsoft/vcpkg ，将开启vcpkg下载；
3、下载完成后，会自动建立一个vcpkg文件夹，就需要再次使用 cd 命令到vcpkg文件路径内；

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201025162415422.png#pic_center)

4、运行 .\bootstrap-vcpkg.bat 等待运行完成。至此vcpkg安装完成。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201025162638194.png#pic_center)

### 5、Visual Studio与链接：

为了在Visual Studio中使用vcpkg，只需要进行一下操作，在之后的使用中就可以直接使用了（需要开启管理员权限），在完成以上操作的前提下，运行命令： .\vcpkg integrate install。运行结果如下，即链接成功，系统里所有C++编辑器都能与vcpkg建立链接了。

提示需要cmake,这个在之后安装库时会自动下载安装。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201025164122903.png#pic_center)

### 6、C\C++库的安装：

运行命令： .\vcpkg install [packages to install] 。
每次安装时，打开Git CMD后只需要cd到vcpkg的安装目录即可（如：C:\src\vcpkg）。使用 vcpkg help 或者 vcpkg help[comment] 可以获取特定命令的命令帮助，通过。.\vcpkg search [search term] 。
注意： vcpkg默认安装编译的是 X86 ，可以通过命令指定为 X64 ，例如：

.\vcpkg install zlib:x64-windows
.\vcpkg install zlib openssl --triplet x64-windows

### 7、C\C++库的卸载：

运行命令：.\vcpkg remove [packages to remove]

### 8、vcpkg优点：

可以对库进行编译，使得各种库的版本同一，不会出现在调用各种第三方库时出现版本不统一问题，同时在出现版本问题时可以使用 vcpkg list 查看已安装的库版本，然后直接进行对应的更新操作，十分简便。

### 9、C\C++常用命令：

集成到全局：vcpkg integrate install
移除全局：vcpkg integrate remove
集成到工程：vcpkg integrate project（在“\scripts\buildsystems”目录下，生成nuget配置文件）
查看库目录：vcpkg search
查看支持的架构：vcpkg help triplet
指定编译某种架构的程序库：vcpkg install xxxx:x64-windows（x86-windows）
卸载已安装库：vcpkg remove xxxx
指定卸载平台：vcpkg remove xxxx:x64-windows
移除所有旧版本库：vcpkg remove --outdated
查看已经安装的库：vcpkg list
更新已经安装的库：vcpkg update xxx
导出已经安装的库：vcpkg export xxxx --7zip（–7zip –raw –nuget –ifw –zip）

