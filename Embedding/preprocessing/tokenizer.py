import javalang
import os
import codecs
import collections
import pickle
import fnmatch

vocabulary = []
vocabulary_size = 100000
max = 1000000

def tokenize(file):
    global vocabulary
    lines = file.read()
    try:
        tokens = javalang.tokenizer.tokenize(lines)
        for token in tokens:
            vocabulary.append(token.value)
    except javalang.tokenizer.LexerError:
        print("Could not process " + file.name + "\n" + "Most likely are not Java")

def build_dict():
    global vocabulary
    global vocabulary_size
    count = [["UNK", -1]]
    count.extend(collections.Counter(vocabulary).most_common(vocabulary_size-1))
    dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)
    unk_count = 0
    for word in vocabulary:
        index = dictionary.get(word, 0)
        if(index == 0):
            unk_count += 1
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return count,dictionary,reverse_dictionary

def file_to_ind(file,dictionary,index_path):
    lines = file.read()
    current_line = 1
    try:
        indexs = []
        tokens = javalang.tokenizer.tokenize(lines)
        for token in tokens:
            indexs.append(dictionary.get(token.value, 0))
        with open(index_path, "wb") as f:
            pickle.dump(indexs, f)
    except javalang.tokenizer.LexerError:
        print("Could not process " + file.name + "\n" + "Most likely are not Java")

def main():
    global vocabulary
    global vocabulary_size
    global max

    dir_path = "PATH_TO_JAVA_PROJECTS"

    print("Building dictionary")

    count = 0
    for root, dirnames, filenames in os.walk(dir_path):
        for filename in fnmatch.filter(filenames, '*.java'):
            try:
                with codecs.open(os.path.join(root, filename), "r", encoding="UTF-8") as file:
                    tokenize(file)
                    count += 1
                if(count % 100 == 0):
                    print("Counting tokens, total: " + str(count) + " files")
                if(count >= max):
                    break
            except UnicodeDecodeError:
                try:
                    with codecs.open(os.path.join(root, filename), "r", encoding="ISO-8859-1") as file:
                        tokenize(file)
                        count += 1
                    if(count % 100 == 0):
                        print("Counting tokens, total: " + str(count) + " files")
                    if(count >= max):
                        break
                except UnicodeDecodeError:
                    print("Unkown encoding: " + os.path.join(root, filename))
                except:
                    pass
            except:
                pass

        if(count >= max):
            break

    print("Building dictionary")
    count,dictionary,reverse_dictionary = build_dict()

    print('Most common words (+UNK)', count[:10])

    count = 0
    index_path = "PATH_TO_STORE_INDEXED_PROEJCTS"
    for root, dirnames, filenames in os.walk(dir_path):
        for filename in fnmatch.filter(filenames, '*.java'):
            try:
                with codecs.open(os.path.join(root, filename), "r", encoding="UTF-8") as file:
                    file_to_ind(file, dictionary, index_path+str(count)+".pickle")
                    count += 1
                if(count % 100 == 0):
                    print("Indexed " + str(count) + " files")
                if(count >= max):
                    break
            except UnicodeDecodeError:
                try:
                    with codecs.open(os.path.join(root, filename), "r", encoding="ISO-8859-1") as file:
                        file_to_ind(file, dictionary, index_path+str(count)+".pickle")
                        count += 1
                    if(count % 100 == 0):
                        print("Indexed " + str(count) + " files")
                    if(count >= max):
                        break
                except UnicodeDecodeError:
                    print("Unkown encoding: " + os.path.join(root, filename))
                except:
                    pass
            except:
                pass

        if(count >= max):
                break

    print("Saving count, dictionary, reverse_dictionary and vocabulary_size")

    with open("PATH_TO_STORE_THE_DICTIONARY" , "wb") as f:
        pickle.dump([count,dictionary,reverse_dictionary,vocabulary_size],f)

    print("Done")

if __name__=="__main__":
    main()
