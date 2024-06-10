import math
import re
import os
import shutil
from sctokenizer import CTokenizer, JavaTokenizer, TokenType

def tokenize_code(code):
    """
    Tokenize Java code using JavaTokenizer from sctokenizer.

    Parameters:
    code (str): The Java code to tokenize.

    Returns:
    list: A list of token types or token values.
    """

    tokenizer = JavaTokenizer()
    tokens = tokenizer.tokenize(code)
    token_list = []
    for token in tokens:
      if token.token_type == TokenType.IDENTIFIER:
        token_list.append(token.token_type)
      else:
        token_list.append(token.token_value)

    return token_list

import re

class Document:
  """
  Represents a document object for storing and processing text content.
  """
  all_documents = []

  def __init__(self, doc_name: str) -> None:
    """
    Initializes a Document object with the provided document name.

    Args:
        doc_name (str): The name of the document file.
    """
    self.doc_name = doc_name
    self.word_dict: dict[str, int] = {}
    self.raw_text = ""
    self.text = ""
    self.tokens: list[str] = []
    self.raw_comments: str = ""
    self.token_comments: list[str] = []

    self.__read_text()
    self.__create_word_dict()
    self.__get_comments(self.raw_text)
    Document.all_documents.append(self)

  def __get_comments(self, code):
    """
    Extracts comments from the provided code string.

    Args:
        code (str): The code string to extract comments from.
    """
    single_line_comment_pattern = r'//.*'
    multi_line_comment_pattern = r'/\*[\s\S]*?\*/'

    single_line_comments = re.findall(single_line_comment_pattern, code)
    multi_line_comments = re.findall(multi_line_comment_pattern, code)

    stripped_single_line_comments = [comment.lstrip('//').strip() for comment in single_line_comments]
    stripped_multi_line_comments = [re.sub(r'(^/\*|\*/$)', '', comment).strip() for comment in multi_line_comments]

    all_comments = ""
    for comment in stripped_single_line_comments:
      all_comments += f"{comment} "

    for comment in stripped_multi_line_comments:
      all_comments += f"{comment}"

    self.raw_comments = all_comments
    self.token_comments = self.raw_comments.split(" ")

  def __read_text(self) -> None:
    """
    Reads the text content from the document file.
    """
    raw_text = ""
    text = ""
    file = open(f"{self.doc_name}", "r", encoding="utf-8")
    while True:
      line = file.readline()
      if not line:
        break

      raw_text += line
      tokenized_line = tokenize_code(line)
      for token in tokenized_line:
        self.tokens.append(token)

      text += " ".join(str(tokenized_line))
    file.close()

    self.raw_text = raw_text
    self.text = text

  def __create_word_dict(self):
    """
    Creates a dictionary to store the frequency of each word in the document.
    """
    for token in self.tokens:
      self.word_dict[token] = self.word_dict.get(token, 0) + 1

class Compare:
  """
  Compares documents using TF-IDF (Term Frequency-Inverse Document Frequency).
  """
  word_doc_freq: dict[str, tuple[list[int], set[str]]] = {}
  def __init__(self):
    """
    Initializes a Compare object. Sets the number of documents to be compared (default 2).
    """
    self.__n_docs = 2

  def __get_global_dict(self, doc_1: Document, doc_2: Document) -> None:
    """
    Creates a dictionary to store word frequencies for both documents.

    Args:
        doc_1 (Document): The first document to compare.
        doc_2 (Document): The second document to compare.
    """
    Compare.word_doc_freq = {}
    for word in doc_1.word_dict:
      Compare.word_doc_freq[word] = ([doc_1.word_dict[word]], set([doc_1.doc_name]))

    for word in doc_2.word_dict:
      if word not in Compare.word_doc_freq:
        Compare.word_doc_freq[word] = ([doc_2.word_dict[word]], set([doc_2.doc_name]))
      else:
        Compare.word_doc_freq[word][0][0] += doc_2.word_dict[word]
        Compare.word_doc_freq[word][1].add(doc_2.doc_name)

  def __calc_idf(self) -> list[float]:
    """
    Calculates Inverse Document Frequency (IDF) for each word in the vocabulary.

    Returns:
        list[float]: A list containing IDF values for all words.
    """
    list_idf = []
    for word in Compare.word_doc_freq:
      idf = math.log((self.__n_docs) / (len(Compare.word_doc_freq[word][1]) + 1)) + 1
      list_idf.append(idf)

    return list_idf

  def __generate_tf(self, doc: Document) -> list[float]:
    """
    Calculates Term Frequency (TF) for each word in a document.

    Args:
        doc (Document): The document to calculate TF for.

    Returns:
        list[float]: A list containing TF values for all words in the document.
    """
    doc_tf: list[float] = []
    n_words: int = sum(doc.word_dict.values())

    for word in Compare.word_doc_freq:
      if word not in doc.word_dict:
        doc_tf.append(0)
      else:
        doc_tf.append(doc.word_dict[word] / n_words)

    return doc_tf

  def __calc_tf_idf(self, tf: list[float], idf: list[float]) -> list[float]:
    """
    Calculates TF-IDF (Term Frequency-Inverse Document Frequency) for each word.

    Args:
        tf (list[float]): A list containing TF values.
        idf (list[float]): A list containing IDF values.

    Returns:
        list[float]: A list containing TF-IDF values for all words.
    """
    tf_idf: list[float] = []
    for i in range(len(idf)):
      tf_idf.append(tf[i] * idf[i])

    return tf_idf

  @staticmethod
  def calc_dot_product(u_vector: list[float], v_vector: list[float]) -> float:
    """
    Calculates the dot product of two vectors.

    Args:
        u_vector (list[float]): The first vector.
        v_vector (list[float]): The second vector.

    Returns:
        float: The dot product of the two vectors.

    Raises:
        Exception: If the vectors have different lengths.
    """
    if len(u_vector) != len(v_vector):
      raise Exception("Length of vectors is not equal")

    product: float = 0
    for i in range(len(u_vector)):
      product += u_vector[i] * v_vector[i]
    return product

  @staticmethod
  def calc_magnitude(vector: list[float]) -> float:
    """
    Calculates the magnitude (length) of a vector.

    Args:
        vector (list[float]): The vector to calculate the magnitude for.

    Returns:
        float: The magnitude of the vector.
    """
    magnitude = 0
    for num in vector:
      magnitude += num ** 2
    return math.sqrt(magnitude)

  def get_word_vectors(self, doc_1: Document, doc_2: Document) -> list[list[float]]:
    """
    Generates TF-IDF word vectors for the two documents.

    Args:
        doc_1 (Document): The first document.
        doc_2 (Document): The second document.

    Returns:
        list[list[float]]: A list containing TF-IDF word vectors for both documents.
    """
    self.__get_global_dict(doc_1, doc_2)
    doc_1_tf: list[float] = self.__generate_tf(doc_1)
    doc_2_tf: list[float] = self.__generate_tf(doc_2)
    idf = self.__calc_idf()
    doc_1_tf_idf: list[float] = self.__calc_tf_idf(doc_1_tf, idf)
    doc_2_tf_idf: list[float] = self.__calc_tf_idf(doc_2_tf, idf)

    len_doc_1 = len(doc_1_tf_idf)
    len_doc_2 = len(doc_2_tf_idf)

    if len_doc_1 != len_doc_2:
      if len_doc_1 > len_doc_2:
        diff = len_doc_1 - len_doc_2
        for i in range(diff):
          doc_2_tf_idf.append(0)
      else:
        diff = len_doc_2 - len_doc_1
        for i in range(diff):
          doc_1_tf_idf.append(0)

    return [doc_1_tf_idf, doc_2_tf_idf]

  def compare_docs(self, doc_1: Document, doc_2: Document) -> float:
    """
    Compares the similarity of two documents using cosine similarity.

    Args:
        doc_1 (Document): The first document.
        doc_2 (Document): The second document.

    Returns:
        float: The cosine similarity score between the documents (0.0 to 1.0).
    """
    doc_1_tf_idf, doc_2_tf_idf = self.get_word_vectors(doc_1, doc_2)

    product: float = Compare.calc_dot_product(doc_1_tf_idf, doc_2_tf_idf)
    doc_1_magn: float = Compare.calc_magnitude(doc_1_tf_idf)
    doc_2_magn: float = Compare.calc_magnitude(doc_2_tf_idf)

    similarity: float = product / (doc_1_magn * doc_2_magn)

    return round(similarity, 4)

import math

class Matrix:
  """
  Provides static methods for performing matrix operations.
  """
  @staticmethod
  def transpose_matrix(matrix: list[list[float]]) -> list[list[float]]:
    """
    Transposes a given matrix.

    Args:
        matrix (list[list[float]]): The matrix to transpose.

    Returns:
        list[list[float]]: The transposed matrix.
    """
    new_matrix: list[list[float]] = []
    for col in range(len(matrix[0])):
      new_row = []
      for row in range(len(matrix)):
        new_row.append(matrix[row][col])
      new_matrix.append(new_row)

    return new_matrix

  @staticmethod
  def trace_matrix(matrix: list[list[float]]) -> float:
    """
    Calculates the trace of a square matrix.

    Args:
        matrix (list[list[float]]): The square matrix to calculate the trace for.

    Returns:
        float: The trace of the matrix (sum of diagonal elements).

    Raises:
        ValueError: If the input matrix is not square.
    """
    trace: float = 0
    row: int = 0
    col: int = 0
    for _ in range(len(matrix)):
      trace += matrix[row][col]
      row += 1
      col += 1
    return trace

  @staticmethod
  def equalize_matrixes(matrix_a: list[list[float]], matrix_b: list[list[float]]) -> list[list[list[float]]]:
    """
    Equalizes the dimensions of two matrices by padding with zeros if necessary.

    Args:
        matrix_a (list[list[float]]): The first matrix.
        matrix_b (list[list[float]]): The second matrix.

    Returns:
        list[list[list[float]]]: A list containing the equalized matrices.
    """
    equalized_matrixes: list[list[list[float]]] = []
    matrix_a_col_len: int = len(matrix_a[0])
    matrix_b_row_len: int = len(matrix_b)

    if matrix_a_col_len > matrix_b_row_len:
      diff: int = matrix_a_col_len - matrix_b_row_len
      row_zeroes: list[float] = [0] * matrix_a_col_len
      col_zeroes: list[float] = [0] * diff

      for row in matrix_b:
        row += col_zeroes

      for _ in range(diff):
        matrix_b.append(row_zeroes)

    else:
      diff: int = matrix_b_row_len - matrix_a_col_len
      row_zeroes: list[float] = [0] * matrix_b_row_len
      col_zeroes: list[float] = [0] * diff

      for row in matrix_a:
        row = row + col_zeroes

      for _ in range(diff):
        matrix_a.append(row_zeroes)

    equalized_matrixes = [matrix_a, matrix_b]

    return equalized_matrixes


  @staticmethod
  def multiply_matrix(matrix_a: list[list[float]], matrix_b: list[list[float]]) -> list[list[float]]:
    """
    Multiplies two matrices.

    Args:
        matrix_a (list[list[float]]): The first matrix.
        matrix_b (list[list[float]]): The second matrix.

    Returns:
        list[list[float]]): The resulting product matrix.

    Raises:
        ValueError: If the inner dimensions of the matrices are not compatible for multiplication.
    """
    matrix_c: list[list[float]] = []
    row_a: int = 0

    if len(matrix_a[0]) != len(matrix_b):
      matrix_a, matrix_b = Matrix.equalize_matrixes(matrix_a, matrix_b)

    for row_a in range(len(matrix_a)):
      new_row: list[float] = []
      for col_b in range(len(matrix_b[0])):
        new_val: float = 0
        for col_a in range(len(matrix_a[0])):
          new_val += matrix_a[row_a][col_a] * matrix_b[col_a][col_b]
        new_row.append(new_val)
      matrix_c.append(new_row)

    return matrix_c

  @staticmethod
  def normalize_matrix(matrix: list[list[float]]) -> float:
    """
    Calculates the Frobenius norm of a matrix.

    Args:
        matrix (list[list[float]]): The matrix to calculate the norm for.

    Returns:
        float: The Frobenius norm of the matrix.
    """
    matrix_t = Matrix.transpose_matrix(matrix)
    matrix_c = Matrix.multiply_matrix(matrix_t, matrix)
    trace = Matrix.trace_matrix(matrix_c)

    return math.sqrt(trace)

  @staticmethod
  def print_matrix(matrix: list[list[float]]) -> None:
    """
    Prints a matrix in a formatted way.

    Args:
        matrix (list[list[float]]): The matrix to print.
    """
    print("--------------------------------------------")
    for row in range(len(matrix)):
      for col in range(len(matrix[0])):
        print(f"{round(matrix[row][col], 4)} |", end="")
      print("\n")
    print("--------------------------------------------")

class Markov_Chain:
  """
  Represents a Markov chain for text generation based on a document.
  """
  def __init__(self, doc_name: str, tokenize: bool = False):
    """
    Initializes a MarkovChain object.

    Args:
        doc_name (str): The name of the document to build the chain from.
        tokenize (bool, optional): Whether to tokenize the document
            before processing. Defaults to False.
    """
    self.markov_chain: list[list[float]] = []
    self.doc_name = doc_name
    self.text = ""
    self.tokens: list[str] = []
    self.token_transitions: dict[str, dict[str, int]] = {}

    if tokenize:
      self.__tokenize_file()
    else:
      self.__read_file()

    self.__generate_token_transitions()
    self.__generate_markov_chain()

  def __read_file(self):
    """
    Reads the text content from the document file.
    """
    text = ""
    file = open(f"{self.doc_name}", "r", encoding="utf-8")
    while True:
      line = file.readline()

      if not line:
        break

      tokenized_line = tokenize_code(line)
      for token in tokenized_line:
        self.tokens.append(token)
      text += " ".join(str(tokenized_line))
    file.close()

    self.text = text

  def __generate_token_transitions(self):
    """
    Generates a dictionary representing token transitions in the document.
    """
    for i, token in enumerate(self.tokens):
      if token not in self.token_transitions:
        self.token_transitions[token] = {}

      if i < len(self.tokens) - 1 and self.tokens[i + 1] not in self.token_transitions[token]:
        self.token_transitions[token][self.tokens[i + 1]] = 1
      elif i < len(self.tokens) - 1 and self.tokens[i + 1] in self.token_transitions[token]:
        self.token_transitions[token][self.tokens[i + 1]] += 1

  def __generate_markov_chain(self):
    """
    Generates the Markov chain transition matrix from token transitions.
    """
    self.markov_chain = []

    for key in self.token_transitions:
      row: list[float] = []
      total_transitions: int = 0
      for freq in self.token_transitions[key].values():
        total_transitions += freq
      for next_key in self.token_transitions:
        if next_key not in self.token_transitions[key]:
          row.append(0)
        else:
          row.append(self.token_transitions[key][next_key] / total_transitions)
      self.markov_chain.append(row)

def cosine_similarity(matrix_a: list[list[float]], matrix_b: list[list[float]]):
  """
  Calculates the cosine similarity between two matrices.

  Args:
      matrix_a (list[list[float]]): The first matrix.
      matrix_b (list[list[float]]): The second matrix.

  Returns:
      float: The cosine similarity between the matrices (0.0 to 1.0).
  """
  norm_matrix_a = Matrix.normalize_matrix(matrix_a)
  norm_matrix_b = Matrix.normalize_matrix(matrix_b)
  matrix_bt = Matrix.transpose_matrix(matrix_b)
  matrix_c = Matrix.multiply_matrix(matrix_bt, matrix_a)
  trace = Matrix.trace_matrix(matrix_c)

  return round(trace / (norm_matrix_a * norm_matrix_b), 4)


def euclidean(vector1, vector2):
    """
    Calculate the Euclidean distance between two vectors.

    Parameters:
    vector1 (list): The first vector.
    vector2 (list): The second vector.

    Returns:
    float: The Euclidean distance between the two vectors.

    Raises:
    ValueError: If the vectors are not of the same length.
    """
    if len(vector1) != len(vector2):
        raise ValueError("Vectors should have the same length")

    sum_squares = sum((a - b) ** 2 for a, b in zip(vector1, vector2))
    distance = math.sqrt(sum_squares)

    return distance

def manhattan(vector1, vector2):
    """
    Calculate the Manhattan distance between two vectors.

    Parameters:
    vector1 (list): The first vector.
    vector2 (list): The second vector.

    Returns:
    float: The Manhattan distance between the two vectors.

    Raises:
    ValueError: If the vectors are not of the same length.
    """
    if len(vector1) != len(vector2):
        raise ValueError("Vectors should have the same length")
    distance = sum(abs(a - b) for a, b in zip(vector1, vector2))

    return distance

def jaccard(doc_1_tokens: Document, doc_2_tokens: Document):
    """
    Calculates the Jaccard similarity between two sets of tokens.

    Args:
        doc_1_tokens (set[str]): The set of tokens from the first document.
        doc_2_tokens (set[str]): The set of tokens from the second document.

    Returns:
        float: The Jaccard similarity score between the documents (0.0 to 1.0).
    """
    tokens1 = set(doc_1_tokens)
    tokens2 = set(doc_2_tokens)

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    similarity = len(intersection) / len(union)

    return similarity

def space_new_line_similarity(code1, code2):
    """
    Calculate the similarity between two pieces of code based on spaces, tabs, and newlines.

    Parameters:
    code1 (str): The first piece of code.
    code2 (str): The second piece of code.

    Returns:
    float: The similarity score between the two pieces of code.
    """
    tabs_distance = abs(code1.count('\t') - code2.count('\t'))
    spaces_distance = abs(code1.count(' ') - code2.count(' '))
    newlines_distance = abs(code1.count('\n') - code2.count('\n'))

    total_tabs = max(code1.count('\t'), code2.count('\t'))
    total_spaces = max(code1.count(' '), code2.count(' '))
    total_newlines = max(code1.count('\n'), code2.count('\n'))

    ED = tabs_distance + spaces_distance + newlines_distance
    total = total_tabs + total_spaces + total_newlines
    if total == 0:
        SNS = 1.0  # Avoid division by zero; assume perfect similarity if no spaces, tabs, or newlines
    else:
        SNS = 1 - ED / total
    return SNS

def classify_braces(code):
    """
    Classify braces in the given code and return a string representation.

    Parameters:
    code (str): The code to classify braces.

    Returns:
    str: A string representing the classification of braces in the code.
    """
    brace_notation = []
    lines = code.split('\n')
    for line in lines:
        stripped = line.strip()
        if '{' in stripped or '}' in stripped:
            if stripped.startswith('{') or stripped.startswith('}'):
                if len(stripped) > 1:
                    brace_notation.append('1')
                else:
                    brace_notation.append('4')
            elif stripped.endswith('{') or stripped.endswith('}'):
                brace_notation.append('2')
            else:
                brace_notation.append('3')
    return ''.join(brace_notation)

def lcs_length(s1, s2):
    """
    Calculate the length of the longest common subsequence (LCS) between two strings.

    Parameters:
    s1 (str): The first string.
    s2 (str): The second string.

    Returns:
    int: The length of the LCS.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[m][n]

def calculate_brace_similarity(code1, code2):
    """
    Calculate the similarity between two pieces of code based on their brace notation.

    Parameters:
    code1 (str): The first piece of code.
    code2 (str): The second piece of code.

    Returns:
    float: The similarity score between the two pieces of code.
    """
    notation1 = classify_braces(code1)
    notation2 = classify_braces(code2)

    LCS = lcs_length(notation1, notation2)
    L1, L2 = len(notation1), len(notation2)

    if L1 == 0 or L2 == 0:
        return 0.0
    BS = 2 * LCS / (L1 * L2)
    return BS

def code_style_similarity(bs, cs, snl):
  """
  Calculates a code style similarity score based on three metrics.

  Args:
      bs (float): The bracket similarity of the code.
      cs (float): The comment similarity of the code.
      snl (float): The space-newline similarity of the code.

  Returns:
      float: The combined code style similarity score (average of the three inputs).
  """
  return (bs + cs + snl) / 3

def table_generator():
  """
  Generates a table of document comparison metrics for a single pair of code files
  and then deletes the files after processing.

  Returns:
      list: A list containing a single dictionary with comparison metrics for
            the processed code file pair.
  """

  directorio_principal = 'flask-backend/queries/'
  data = []

  compare = Compare()

  if os.path.exists(directorio_principal):
    archivos = [os.path.join(directorio_principal, archivo) for archivo in os.listdir(directorio_principal)]

    if len(archivos) != 2:
      print(f"Error: Expected exactly two files in {directorio_principal}.")
      return data

    doc1 = Document(archivos[0])
    doc2 = Document(archivos[1])

    doc1Mk = Markov_Chain(archivos[0])
    doc2Mk = Markov_Chain(archivos[1])

    tf_idf = compare.compare_docs(doc1, doc2)
    markov = cosine_similarity(doc1Mk.markov_chain, doc2Mk.markov_chain)
    vector1, vector2 = compare.get_word_vectors(doc1, doc2)
    euc = euclidean(vector1, vector2)
    mht = manhattan(vector1, vector2)
    jac = jaccard(doc1.tokens, doc2.tokens)
    snl = space_new_line_similarity(doc1.raw_text, doc2.raw_text)
    bs = calculate_brace_similarity(doc1.raw_text, doc2.raw_text)
    cs = jaccard(doc1.token_comments, doc2.token_comments)
    style_similarity = code_style_similarity(bs, cs, snl)

    _map = {
        "Tf idf": tf_idf,
        "Markov": markov,
        "Euclidean": euc,
        "Manhattan": mht,
        "Jaccard": jac,
        "Space_NewLine": snl,
        "BraceSimilarity": bs,
        "CommentSimilarity": cs,
        "CodeStyleSimilarity": style_similarity,
    }

    data.append(_map)

    for archivo in archivos:
      os.remove(archivo)
  else:
    print(f'The directory {directorio_principal} does not exist.')

  return data
