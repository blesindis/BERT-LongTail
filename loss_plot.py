import matplotlib.pyplot as plt
# Updated data with the additional values
axis_x = ['0.00 ~ 0.62', '0.62 ~ 1.25', '1.25 ~ 1.88', '1.88 ~ 2.50', '2.50 ~ 3.12', '3.12 ~ 3.75', '3.75 ~ 4.38', '4.38 ~ 5.00', '5.00 ~ 5.62', '5.62 ~ 6.25', '6.25 ~ 6.88', '6.88 ~ 7.50', '7.50 ~ 8.12', '8.12 ~ 8.75', '8.75 ~ 9.38', '9.38 ~ 10.00', '10.00 ~ 10.62', '10.62 ~ 11.25', '11.25 ~ 11.88', '11.88 ~ 12.50', '12.50 ~ 13.12', '13.12 ~ 13.75']
bert_y = [22, 13, 35, 79, 171, 336, 495, 440, 325, 194, 112, 71, 44, 25, 11, 6, 8, 4, 1,0,0,0]
moe_y = [58, 78, 237, 428, 494, 446, 253, 152, 97, 65, 36, 20, 14, 8, 5, 2, 2, 0,0,0,0,0]
tail_v2 = [57, 65, 172, 371, 515, 458, 285, 168, 127, 64, 43, 27, 8, 13, 8, 2, 1, 1, 1, 2, 0, 0]

tail_y = [70, 87, 206, 394, 519, 453, 276, 152, 94, 53, 36, 20, 12, 14, 2, 7, 0, 2, 0,0,0,0]

print(len(bert_y), len(moe_y), len(tail_v2), len(tail_y), len(axis_x))

# Plotting the updated line chart
plt.figure(figsize=(14, 7))  # Larger figure size for readability

# Plotting each line
plt.plot(axis_x, bert_y, label='bert')
plt.plot(axis_x, moe_y, label='moe')
plt.plot(axis_x, tail_v2, label='tail_v2')
plt.plot(axis_x, tail_y, label='tail')

# Improving the x-axis
plt.xticks(rotation=90, ha='center', fontsize=10)  # Rotate to prevent overlap
plt.xlabel('Value Ranges')  # X-axis Label
plt.ylabel('Counts')  # Y-axis Label
plt.title('Comparison of Model Loss Across Value Ranges')  # Chart title
plt.grid(True)  # Add gridlines
plt.legend()  # Add legend to identify lines

plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
plt.savefig("Loss Comparison.png")
