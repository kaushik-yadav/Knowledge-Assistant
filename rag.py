import os
import re

docs = {}
# taking all the text files present in docs
txt_files = [file for file in os.listdir('docs/')  if file.endswith('.txt')]

for file in txt_files:
    with open(os.path.join("docs",file)) as f:
        docs[file.replace(".txt","")] = f.read()

# chunking each document like {source : 'doc category', content : 'the content in category'}
def get_formatted_data(filename):
    with open(filename, 'r') as f:
        data = f.read()
        source = filename.split('.')[0].split('/')[1]

        # splitting the product specs data into seperate product wise data
        if source == 'product_specs':
            split_text = re.split('(?=Product Name)',data)[1:]

        # splitting the company faqs into pairs of Q and A
        # for company overview splitting it paragraph wise
        else: 
            split_text = re.split('\n\n', data)
        return [{'source':source, 'content':x} for x in split_text] 

chunked_data = []

# storing the chunked data of each doc
for file in txt_files:
    chunked_data.extend(get_formatted_data(f'docs/{file}'))

print(chunked_data)