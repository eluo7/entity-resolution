import csv
import random
from faker import Faker
from datetime import datetime

fake = Faker('zh_CN')

# 定义时间段
TIME_SLOTS = [
    "08:00-10:00", "10:00-12:00", "12:00-14:00",
    "14:00-16:00", "16:00-18:00", "18:00-20:00"
]

# 常见寄件物品
COMMON_ITEMS = [
    "文件", "电子产品", "服装", "食品", "日用品",
    "书籍", "化妆品", "玩具", "礼品", "其他"
]

# 生成身份证号
def generate_id_number():
    # 省份代码
    province_code = random.choice([
        '11', '12', '13', '14', '15', '21', '22', '23',
        '31', '32', '33', '34', '35', '36', '37', '41',
        '42', '43', '44', '45', '46', '50', '51', '52',
        '53', '54', '61', '62', '63', '64', '65'
    ])
    
    # 城市和区县代码
    city_code = fake.random_number(digits=2, fix_len=True)
    county_code = fake.random_number(digits=2, fix_len=True)
    
    # 出生日期
    birth_date = fake.date_of_birth(minimum_age=18, maximum_age=70)
    birth_code = birth_date.strftime('%Y%m%d')
    
    # 顺序码
    seq_code = fake.random_number(digits=3, fix_len=True)
    
    # 校验码
    check_code = random.choice(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'X'])
    
    return f"{province_code}{city_code}{county_code}{birth_code}{seq_code}{check_code}"

def generate_customer_data(num_records=100):
    customers = []
    
    for _ in range(num_records):
        customer = {
            "phone_number": fake.phone_number(),
            "email": fake.email(),
            "id_number": generate_id_number(),
            "common_send_time": random.choice(TIME_SLOTS),
            "common_send_item": random.choice(COMMON_ITEMS),
            "common_send_address": fake.address(),
            "avg_send_weight": round(random.uniform(0.5, 10.0), 2),
            "send_frequency": random.choice(["每天", "每周2-3次", "每周1次", "每月2-3次", "每月1次"])
        }
        customers.append(customer)
    
    return customers

def generate_adjusted_data(original_data):
    adjusted_data = []
    
    for customer in original_data:
        adjusted_customer = customer.copy()
        # 微调部分字段
        adjusted_customer["phone_number"] = adjusted_customer["phone_number"][:-1] + str(random.randint(0,9))
        adjusted_customer["email"] = adjusted_customer["email"].replace("@", "_" + str(random.randint(0,9)) + "@")
        adjusted_customer["common_send_address"] = fake.address()
        adjusted_customer["avg_send_weight"] = round(adjusted_customer["avg_send_weight"] * random.uniform(0.9, 1.1), 2)
        adjusted_data.append(adjusted_customer)
    
    return adjusted_data

def save_to_csv(data, filename="customer_data.csv"):
    fieldnames = [
        "phone_number", "email", "id_number", "common_send_time",
        "common_send_item", "common_send_address", "avg_send_weight",
        "send_frequency"
    ]
    
    with open(filename, mode='w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"成功生成 {len(data)} 条客户数据并保存到 {filename}")

if __name__ == "__main__":
    num_records = int(input("请输入要生成的记录数量(默认100): ") or 100)
    filename = input("请输入CSV文件名(默认customer_data.csv): ") or "customer_data.csv"
    adjusted_filename = "adjusted_" + filename
    
    customer_data = generate_customer_data(num_records)
    adjusted_data = generate_adjusted_data(customer_data)
    
    save_to_csv(customer_data, filename)
    save_to_csv(adjusted_data, adjusted_filename)
    print(f"成功生成 {len(customer_data)} 条原始客户数据和 {len(adjusted_data)} 条微调客户数据")