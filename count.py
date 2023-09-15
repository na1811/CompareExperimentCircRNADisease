# def count_ones_in_csv(file_path):
#     try:
#         with open(file_path, 'r') as file:
#             data = file.read()
#             ones_count = data.count('1')
#             return ones_count
#     except FileNotFoundError:
#         print("Không tìm thấy file.")
#         return 0

# file_path = './Data/circRNA_disease_from_circRNADisease/association.csv'  # Thay đổi đường dẫn đến file CSV thực tế
# count = count_ones_in_csv(file_path)
# print("Số lượng số 1 trong file là:", count)


import sys

def count_ones_in_csv(file_path):
    try:
        with open(file_path, 'r') as file:
            data = file.read()
            ones_count = data.count('0')
            return ones_count
    except FileNotFoundError:
        print("Không tìm thấy file.")
        return 0

file_path = './Data/circRNA_disease_from_circRNADisease/association.csv'  # Thay đổi đường dẫn đến file CSV thực tế
count = count_ones_in_csv(file_path)

sys.stdout.buffer.write(f"Số lượng số 0 trong file là: {count}\n".encode('utf-8'))
