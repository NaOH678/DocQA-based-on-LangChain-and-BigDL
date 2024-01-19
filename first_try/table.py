from prettytable import PrettyTable

# 创建表格对象
table = PrettyTable()

# 添加表头
table.field_names = ["Name", "Age", "City"]

# 添加数据行
table.add_row(["Alice", 25, "New York"])
table.add_row(["Bob", 30, "San Francisco"])
table.add_row(["Charlie", 22, "Los Angeles"])

# 输出表格
print(table)