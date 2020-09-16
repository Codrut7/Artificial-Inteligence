from googletrans import Translator

# SHIT DOES NOT FUCKING WORK
# GOOGLE TRANSLATE MEH

def translate_file(PATH, WRITE_PATH):
    with open(PATH, encoding='utf-8') as f:
        lines = f.readlines()
    translator = Translator()
    f = open(WRITE_PATH, "w", encoding='utf-8')

    # Google supports only < 14k line to translate per request
    batch_lines = []
    
    for i, line in enumerate(lines):
        if i < 14000:
            inp, target = line.split('\t')
            inp = inp.replace('\n', '')
            target = target.replace('\n', '')
            batch_lines.append(inp)
            batch_lines.append(target)
        else:
            print("Translating ...")
            translations = translator.translate(batch_lines, src='en', dest='ro')
            for idx in range(len(0, len(translations), 2)):  
                f.write(translations[idx].text + '\t' + translations[idx+1].text + '\n')
            batch_lines = []

    f.close()

PATH = r'C:\Projects\NLP\chatbot\subtitrari\dataset.txt'
WRITE_PATH = r'C:\Projects\NLP\chatbot\subtitrari\translated_dataset.txt'
translate_file(PATH, WRITE_PATH)
