import pandas as pd

all_data = pd.read_csv('pybert/dataset/raw/raw_all.csv',  encoding='ISO-8859-1')

text_data = all_data[['id','NoteText']]
# print(text_data.shape)

text_data = text_data.dropna(axis=0, how='any')

french_letters = ['é',
        'à', 'è', 'ù',
        'â', 'ê', 'î', 'ô', 'û',
        'ç',
        'ë', 'ï', 'ü']

English_letters = ['a','b','c','d','e','f','g', 'h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                   'A','B','C','D','E','F','G', 'H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

for i in range(text_data.shape[0]):
    is_french_exist = True if set(french_letters).intersection(text_data['NoteText'][i]) else False
    no_english = False if set(English_letters).intersection(text_data['NoteText'][i]) else True
    if is_french_exist:
        text_data.iloc[i,1] = 'NaN'
    elif no_english:
        text_data.iloc[i,1] = 'NaN'
text_data = text_data.drop(text_data[text_data['NoteText'] == 'NaN'].index)

text_data = text_data.dropna(axis=0, how='any')
print(text_data.shape)
text_data.to_csv('pybert/dataset/test.csv', index=False)
