import os
import shutil
import csv

data_dir_path = os.path.join(os.path.dirname(__file__), "data")
image_dir_path = os.path.join(data_dir_path, "images")
csv_dir_path = os.path.join(data_dir_path, "csv")
renamed_data_dir_path = os.path.join(os.path.dirname(__file__), "renamed_data")

if not os.path.isdir(renamed_data_dir_path):
    os.makedirs(renamed_data_dir_path)

renamed_csv_dir_path = os.path.join(renamed_data_dir_path, "csv")
if not os.path.isdir(renamed_csv_dir_path):
    os.makedirs(renamed_csv_dir_path)

image_dir_list = os.listdir(image_dir_path)
for char_code in image_dir_list:
    renamed_char_code_dir_path = os.path.join(renamed_data_dir_path, char_code)
    if not os.path.isdir(renamed_char_code_dir_path):
        os.makedirs(renamed_char_code_dir_path)
    char_code_dir_path = os.path.join(image_dir_path, char_code)
    image_file_name_list = os.listdir(char_code_dir_path)
    for image_file_name in image_file_name_list:
        image_file_path = os.path.join(char_code_dir_path, image_file_name)
        splited_image_file_name = os.path.splitext(image_file_name)
        zero_padding_image_number = splited_image_file_name[0].zfill(5)
        renamed_image_file_path = os.path.join(
            renamed_char_code_dir_path, f"{char_code}_{zero_padding_image_number}{splited_image_file_name[1]}")
        shutil.copy(image_file_path, renamed_image_file_path)

    csv_file_path = os.path.join(csv_dir_path, char_code, "times.csv")
    renamed_csv_contents = []
    with open(csv_file_path, "r", encoding="utf_8_sig") as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        renamed_csv_contents.append(header)
        for row in csv_reader:
            renamed_id = f"{char_code}_{row[0].zfill(5)}"
            renamed_csv_contents.append([renamed_id, row[1]])
    renamed_csv_file_path = os.path.join(renamed_csv_dir_path, f"{char_code}.csv")
    with open(renamed_csv_file_path, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(renamed_csv_contents)
