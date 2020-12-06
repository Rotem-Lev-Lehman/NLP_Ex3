import numpy as np

# legal_lines = []
# with open('/content/embedding_stuff/glove/glove.6B.300d.txt') as file1:
#     lines = file1.readlines()
#     for l in lines:
#         split_line = l.rstrip().split()
#         #word = porter.stem(split_line[0])
#         word = split_line[0]
#         if word in legal_words:
#             legal_lines.append(l)
#
# with open('embedded.txt', 'w') as write_file:
#     write_file.writelines(legal_lines)
#

print('Creating embedding dictionary')
embedding_dict = {}
with open('embedded.txt', encoding='utf-8') as file1:
    all_lines = file1.readlines()
    for line in all_lines:
        split_line = line.strip().split()
        embedding_dict[split_line[0]] = np.array(split_line[1:]).astype(np.float32)
print('Done creating embedding dictionary')
