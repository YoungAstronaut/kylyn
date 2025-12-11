from recipe.dapo.embed_utils import find_first_descent_point

if __name__ == "__main__":
    # 测试1：标准下降场景
    data1 = [1.0, 2.0, 3.0, 2.7, 1.5, 1.2, 2.0, 3.0]
    print(f"案例1 - 低点索引: {find_first_descent_point(data1, 0.5)}")  # 返回: 5 (值1.2)

    # 测试2：下降不足（不满足阈值）
    data2 = [1.0, 2.0, 3.0, 2.9, 2.8, 2.7, 3.0]
    print(f"案例2 - 低点索引: {find_first_descent_point(data2, 0.5)}")  # 返回: -1

    # 测试3：多个下降沿（返回第一个满足条件的）
    data3 = [1.0, 3.0, 2.0, 4.0, 1.5, 2.0]
    print(f"案例3 - 低点索引: {find_first_descent_point(data3, 0.8)}")  # 返回: 2 (值2.0)

    # 测试4：含噪声数据
    data4 = [1.0, 1.1, 0.9, 1.05, 3.0, 2.9, 2.8, 1.5, 1.6, 1.4]
    print(f"案例4 - 低点索引: {find_first_descent_point(data4, 1.0)}")  # 返回: 7 (值1.5)
