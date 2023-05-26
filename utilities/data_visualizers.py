




def view_sentence(sentences, phase='', limit=5):
    for sentence in sentences.iloc[:limit]:
        print(f'{phase}phase:\n{sentence}\n')



