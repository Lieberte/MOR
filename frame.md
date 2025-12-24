MOR/
├── CMakeLists.txt                    # 顶层构建配置（统一管理所有模块）
│
├── core/                             # 核心基础设施模块
│   ├── CMakeLists.txt                # 模块独立CMake配置
│   ├── include/
│   │   └── framework/
│   │       └── core/                 # 核心基础设施
│   │           ├── Matrix.h/cpp     # 分布式矩阵类
│   │           ├── Vector.h/cpp     # 分布式向量类
│   │           ├── ParallelContext.h/cpp # 并行上下文管理
│   │           ├── Config.h          # 基础配置类
│   │           └── DataTypes.h       # 通用数据类型定义
│   └── src/                          # 对应源文件
│
├── utils/                            # 工具函数模块
│   └── src/
│
├── mor/                              # MOR模块（模型降阶）
│   ├── CMakeLists.txt                # MOR模块独立CMake配置
│   ├── include/
│   │   └── framework/
│   │       └── mor/                  # MOR命名空间
│   │           ├── algorithms/       # MOR算法
│   │           │   ├── BasisGenerator.h/cpp  # 基向量生成器抽象接口
│   │           │   ├── POD/          # POD
│   │           │   ├── SVD/          # SVD
│   │           │   └── BalancedTruncation/  # Balanced Truncation
│   │           └── io/               
│   │               ├── BasisWriter.h/cpp
│   │               └── BasisReader.h/cpp
│   └── src/                          
│
├── preprocessing/                    # 数据预处理
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── framework/
│   │       └── preprocessing/
│   │           ├── DataCleaner.h/cpp
│   │           ├── Normalizer.h/cpp
│   │           ├── FeatureExtractor.h/cpp
│   │           └── Filter.h/cpp
│   └── src/
│
├── postprocessing/                   # 后处理
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── framework/
│   │       └── postprocessing/
│   │           ├── ErrorAnalyzer.h/cpp
│   │           ├── ResultProcessor.h/cpp
│   │           └── Metrics.h/cpp
│   └── src/
│
├── dataio/                           # 数据I/O
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── framework/
│   │       └── dataio/
│   │           ├── FileReader.h/cpp
│   │           ├── FileWriter.h/cpp
│   │           ├── FormatConverter.h/cpp
│   │           └── DataManager.h/cpp
│   └── src/
│
├── visualization/                    # 可视化
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── framework/
│   │       └── visualization/
│   │           ├── Plotter.h/cpp
│   │           ├── Viewer3D.h/cpp
│   │           └── Exporter.h/cpp
│   └── src/
│
├── hyperreduction/                   # 超约化
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── framework/
│   │       └── hyperreduction/
│   │           ├── DEIM.h/cpp       # Discrete Empirical Interpolation
│   │           ├── QDEIM.h/cpp       # QR-based DEIM
│   │           ├── GNAT.h/cpp        # Gauss-Newton with Approximated Tensors
│   │           └── Sampling.h/cpp    # 采样策略
│   └── src/
│
├── interpolation/                    # 插值
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── framework/
│   │       └── interpolation/
│   │           ├── Interpolator.h/cpp      # 基础插值接口
│   │           ├── MatrixInterpolator.h/cpp # 矩阵插值
│   │           ├── VectorInterpolator.h/cpp # 向量插值
│   │           └── ParametricInterpolator.h/cpp # 参数化插值
│   └── src/
│
├── sampling/                         # 采样
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── framework/
│   │       └── sampling/
│   └── src/
│
└── tests/                            # 单元测试