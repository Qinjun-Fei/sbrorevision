# 导入保存结果的json
import yaml
result_file = '../result/normal/cifar_1009_0812/result_final.json'
with open(result_file, 'r') as f:
    data = yaml.unsafe_load(f)


# 打印读取的数据
pass