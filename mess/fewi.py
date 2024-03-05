import re


# Open the text file for reading
with open(r'C:\Git\NeuralNetwork\Pancreas_with_Monai_Lightning\logs\log_label_check.txt', 'r') as file:
    # Read all lines from the file
    lines = file.readlines()

dict_pre_check = {}

for line in lines:
    # Split the entry by comma and then by colon to isolate pancreas_size

    # Extract the number and pancreas_size using regular expressions
    number_match = re.search(r'number=(\d+)', line)
    pancreas_match = re.search(r'pancreas_size=(\d+)', line)

    if number_match and pancreas_match:
        number = int(number_match.group(1))
        pancreas_size = int(pancreas_match.group(1))

        # Create the dictionary
        dict_pre_check[number] = round(pancreas_size,4)



# print(dict_pre_check)

with open(r'C:\Git\NeuralNetwork\Pancreas_with_Monai_Lightning\logs\pick_data_log.txt', 'r') as file:
    # Read all lines from the file
    lines = file.readlines()

# Initialize an empty dictionary to store the data
dict_predict = {}

# Process each line in the file
for line in lines:
    # Split the line by colon ':' to separate the number and the metric
    pattern = r'number:(\d+) Dice Metric:([\d.]+)'

    # Search for the pattern in the text data
    match = re.search(pattern, line)

    # If the pattern is found
    if match:
        # Extract the number and the Dice Metric from the match object
        number = int(match.group(1))
        dice_metric = round(float(match.group(2)), 4)

        # Create the dictionary
        dict_predict[number] = dice_metric


# Print the resulting dictionary
# print(dict_predict)
predict_low = []
check_low = []
for key,value in dict_predict.items():
    if value < 0.6:
        predict_low.append(key)

for key,value in dict_pre_check.items():
    if value < 10000:
        check_low.append(key)

both = [i for i in predict_low if i in check_low]

print(f"Predict low: {predict_low}")
print(f"Check low: {check_low}")
print(f"Both low: {both}")

for i in check_low:
    print(f"{i}  Predict: {dict_predict[i]} Check: {dict_pre_check[i]}\n")
