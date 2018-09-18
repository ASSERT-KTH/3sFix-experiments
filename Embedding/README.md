The `preprocessing` folder stores `tokenizer.py` to tokenize the directory containing all the Java projects.
The `word2vec` folder stores `word2vec.py` to learn the embedding from the tokenized Java projects.

All path variable in both file need to be changed:
* `PATH_TO_JAVA_PROJECTS` = path to Java projects
* `PATH_TO_STORE_INDEXED_PROEJCTS` = path to indexed Java projects
* `PATH_TO_STORE_THE_DICTIONARY` = path to the dictionary

You are welcome to change other variables such as `vocabulary_size` in `tokenizer.py`, or `num_steps` in `word2vec.py`.
