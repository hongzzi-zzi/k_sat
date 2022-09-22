#%% impotr package

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# import json
import os
from difflib import SequenceMatcher
from string import whitespace

# nltk.download('averaged_perceptron_tagger')
import benepar
import scipy
from nltk import tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from opencc import OpenCC
from stanfordcorenlp import StanfordCoreNLP
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.utils import get_stop_words
from wordhoard import Antonyms, Synonyms

benepar_parser = benepar.Parser("benepar_en3")

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from googletrans import Translator
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import AutoModelWithLMHead, AutoTokenizer

gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_model = AutoModelWithLMHead.from_pretrained("gpt2", pad_token_id=gpt2_tokenizer.eos_token_id)
bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
kw_model = KeyBERT()
##########
# translator = Translator()

gpt2_model.to(DEVICE)
bert_model.to(DEVICE)

#######
from translate import Translator

# In [1]: from translate import Translator
# In [2]: translator= Translator(to_lang="zh")
# In [3]: translation = translator.translate("This is a pen.")
# Out [3]: 这是一支笔

#%%
PATH='/home/stanford-corenlp-4.5.0'

os.chdir(PATH)
os.system('pwd')
os.system('java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
                    -preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
                    -status_port 9000 -port 9009 -timeout 15000 &')
os.chdir('/home/K_sat_final')
nlp = StanfordCoreNLP('http://localhost', port=9009, memory='8g')
cc = OpenCC('t2s')

atom_end = set('()"\'') | set(whitespace)
#%%
def summary(passage:str, num_sentences:int)->list:
    '''
    1, 2, 4, 7, 8, 9
    passage에서 num_sentences개의 중심 문장을 list로 리턴
    '''
    parser = PlaintextParser.from_string(passage, Tokenizer('english'))
    stemmer = Stemmer('english')
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words('english')

    summary=[]

    for sentence in summarizer(parser.document, num_sentences):
        summary.append(str(sentence))
    return summary

def paraphrasing_by_transe(summary:list, midpoint='zh-cn')->list:
    '''
    1, 2, 4, 7, 8
    summart: list
    원래 passage에 없는 문장들로 바꾸어 list로 리턴
    '''
    pharaphrase=[]
    translator= Translator(to_lang="zh")
    for sentence in summary:
        # translate=translator.translate(sentence, src='en', dest=midpoint).text
        translate=translator.translate(sentence)
        # pharaphrase.append(translator.translate(translate, src=midpoint, dest='en').text)
        pharaphrase.append(translator.translate(translate))
    return pharaphrase

def transe_kor(sentence:list)->list:
    '''
    1, 2
    한국어로 번역
    '''
    kor=[]
    translator= Translator(to_lang="kr")
    for sent in sentence:
        # print(type(sent))
        # k_sentence=translator.translate(sent, src='en', dest='ko').text
        k_sentence=translator.translate(sent)
        kor.append(k_sentence)        
    return kor

def sort_by_similarity_sentence(original_sentence:str, generated_sentences_list:list)->list:
    '''
    generate_sentences() 내부에서 쓰임
    '''
    sentence_embeddings = bert_model.encode(generated_sentences_list)
    queries = [original_sentence]
    query_embeddings = bert_model.encode(queries)
    number_top_matches = len(generated_sentences_list)
    dissimilar_sentences = []

    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        for idx, distance in reversed(results[0:number_top_matches]):
            score = 1-distance
            # print(score)
            if score < 0.99:
                dissimilar_sentences.append(generated_sentences_list[idx].strip())
           
    sorted_dissimilar_sentences = sorted(dissimilar_sentences, key=len)
    return sorted_dissimilar_sentences[:2]

def generate_sentences(partial_sentence:str,full_sentence:str)->list:
    '''
    1, 7
    partial_sentence 로 가짜 문장 생성하고
    full_sentence와 비교해서 유사도 높은 문장 3개 리스트로 리턴
    '''
    input_ids = gpt2_tokenizer.encode(partial_sentence, return_tensors='pt') # use tokenizer to encode
    input_ids = input_ids.to(DEVICE)
    maximum_length = len(partial_sentence.split())+80 

    sample_outputs = gpt2_model.generate( 
        input_ids,
        do_sample=True,
        max_length=maximum_length, 
        top_p=0.90, 
        top_k=50,   
        repetition_penalty  = 10.0,
        num_return_sequences=5
    )
    generated_sentences=[]
    for i, sample_output in enumerate(sample_outputs):
        decoded_sentences = gpt2_tokenizer.decode(sample_output, skip_special_tokens=True)
        decoded_sentences_list =tokenize.sent_tokenize(decoded_sentences)
        generated_sentences.append(decoded_sentences_list[0]) # takes the first sentence 
        
    top_3_sentences = sort_by_similarity_sentence(full_sentence, generated_sentences)
    return top_3_sentences

def get_flattened(t)->str:
    '''
    get_right_most_VP_or_NP() 내부에서 쓰임
    '''
    sent_str_final = None
    if t is not None:
        sent_str = [" ".join(x.leaves()) for x in list(t)]
        sent_str_final = [" ".join(sent_str)]
        sent_str_final = sent_str_final[0]
    return sent_str_final

def get_termination_portion(main_string:str,sub_string:str)->str:
    '''
    get_sentence_completions() 내부에서 쓰임
    '''
    combined_sub_string = sub_string.replace(" ","")
    main_string_list = main_string.split()
    last_index = len(main_string_list)
    for i in range(last_index):
        check_string_list = main_string_list[i:]
        check_string = "".join(check_string_list)
        check_string = check_string.replace(" ","")
        if check_string == combined_sub_string:
            return " ".join(main_string_list[:i])       
    return None

def get_right_most_VP_or_NP(parse_tree,last_NP = None,last_VP = None):
    '''
    get_sentence_completions() 내부에서 쓰임
    '''
    if len(parse_tree.leaves()) == 1:
        return get_flattened(last_NP),get_flattened(last_VP)
    last_subtree = parse_tree[-1]
    if last_subtree.label() == "NP":
        last_NP = last_subtree
    elif last_subtree.label() == "VP":
        last_VP = last_subtree
    
    return get_right_most_VP_or_NP(last_subtree,last_NP,last_VP)

def get_sentence_completions(key_sentences:list)->dict:
    '''
    1, 7
    key_sentences list를 받아서 빈칸뚫린? 아무튼 딕셔너리 리턴
    '''
    sentence_completion_dict = {}
    for individual_sentence in key_sentences:
        sentence = individual_sentence.rstrip('?:!.,;')
        tree = benepar_parser.parse(sentence)
        last_nounphrase, last_verbphrase =  get_right_most_VP_or_NP(tree)
        phrases= []
        if last_verbphrase is not None:
            verbphrase_string = get_termination_portion(sentence,last_verbphrase)
            if verbphrase_string is not None:
                phrases.append(verbphrase_string)
                
        if last_nounphrase is not None:
            nounphrase_string = get_termination_portion(sentence,last_nounphrase)
            if nounphrase_string is not None:
                phrases.append(nounphrase_string)
    
        longest_phrase =  sorted(phrases, key=len, reverse=True)
        if len(longest_phrase) == 2:
            first_sent_len = len(longest_phrase[0].split())
            second_sentence_len = len(longest_phrase[1].split())
            if (first_sent_len - second_sentence_len) > 4:
                del longest_phrase[1]
                
        if len(longest_phrase)>0:
            sentence_completion_dict[sentence]=longest_phrase

    return sentence_completion_dict

def get_keyword_list(passage, max_word_cnt:int,top_n:int, option=None)->list:
    '''
    1, 2, 4, 5, 8
    '''
    result=[]
    if type(passage)==list:
        for sentence in passage:
            kwd=kw_model.extract_keywords(sentence, keyphrase_ngram_range=(1,max_word_cnt), stop_words='english', top_n=top_n)
            for i in range(len(kwd)):   ## 점수 빼고 단어만
                kwd[i]=kwd[i][0]
            if option==None:result.append(kwd)
            elif option=='Q8':result.append(kwd)
        # print(result)
        if option=='Q8':
            tmp=1;tmp_list=[]
            for a in result[0]:
                for b in result[1]:
                    similarity=word_similarity(a, b)
                    # print(a, b, similarity)
                    if similarity<tmp:
                        tmp=similarity; tmp_list=[a, b]  
            # print(tmp_list)
            result=tmp_list
                        
    elif type(passage)==str:
        kwd=kw_model.extract_keywords(passage, keyphrase_ngram_range=(1, max_word_cnt), stop_words='english', top_n=top_n)
        for i in range(len(kwd)):   ## 점수 빼고 단어만
                kwd[i]=kwd[i][0]
        # result=kwd
        result.append(kwd)

    return result

def word_similarity(a, b):
    '''
    get_keyword_list(), get_synonym_list(), get_antonym_list() 내부에서 쓰임
    '''
    return SequenceMatcher(None, a, b).ratio()

def get_synonym_list(word:str, num_word:int)->list:
    '''
    4, 8
    word:str  받아서 동의어 num_word개 list로 리턴
    '''
    synonym=Synonyms(search_string= word,max_number_of_requests=15,
                   rate_limit_timeout_period=30)
    synonym_list=synonym.find_synonyms()

    if '\x1b' in synonym_list: 
        index=synonym_list.index('\x1b')
        if index==0:
            print('get_synonym_list: '+word+': can\'t find any synonym')
            return ['None']
        else:
            synonym_list=synonym_list[:index-1]
    elif len(synonym_list)==0:
        return['None']

    tuple_result=[];result=[]
    for synonym in synonym_list:
        similarity=word_similarity(word, synonym)
        tuple_result.append((synonym, similarity))
    sorted(tuple_result, key=lambda tuple_result: tuple_result[1])
    tuple_result=tuple_result[-num_word:]
    for i in tuple_result:
        result.append(i[0])
    return result

def get_antonym_list(word:str, num_word:int)->list:
    '''
    2, 4, 5, 8
    word:str  받아서 반의어 num_word개 list로 리턴
    '''
    antonym=Antonyms(search_string= word,max_number_of_requests=15,
                   rate_limit_timeout_period=30)
    antonym_list=antonym.find_antonyms()
    # print(antonym_list)
    if '\x1b' in antonym_list: 
        index=antonym_list.index('\x1b')
        if index==0:
            print('get_antonym_list: '+word+': can\'t find any antonym')
            return ['None']
        else:
            antonym_list=antonym_list[:index-1]
    elif len(antonym_list)==0:
        return['None']
    tuple_result=[];result=[]
    for antonym in antonym_list:
        similarity=word_similarity(word, antonym)
        tuple_result.append((antonym, similarity))
    sorted(tuple_result, key=lambda tuple_result: tuple_result[1])
    tuple_result=tuple_result[:num_word]
    for i in tuple_result:
        result.append(i[0])
    return result

def del_same_lemmatization(word_list:list)->list:
    '''
    4, 8
    word_list 받아서 같은 원형을 가진 단어 지워진 리스트 반환
    stanfordnlp로 바꿔볼수도..
    '''
    # print(type(word_list))
    if type(word_list)!=list:return None
    lmtzr = WordNetLemmatizer()
    word_list_lemmatize=[]
    for i in word_list:
        word_list_lemmatize.append(lmtzr.lemmatize(i))
    return list(dict.fromkeys(word_list_lemmatize))

## https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/
def pos_tagger(nltk_tag):
    '''
    get_pos() 내부에서 쓰임
    '''
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None

def get_pos(word:str, pos_tagger_flag=True)->str:
    '''
    2, 4, 5, 8
    단어:str 넣으면 포스를 스트링으로 리턴
    stanfordnlp로 바꿔볼수도
    '''
    tagged_list = pos_tag([word])
    if pos_tagger_flag==True:
        wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), tagged_list))
        pos=wordnet_tagged[0][-1]
    else:
        pos=tagged_list[0][-1]

    return pos

def get_lmtzr(word:str, pos=None)->str:## pos: str
    '''
    2, 4, 5, 8
    단어:str, pos(있으면..str) 넣으면 원형 str로 리턴
    stanfordnlp로 바꿔볼수도
    '''
    lmtzr = WordNetLemmatizer()
    if pos==None:     
        lmtzr=lmtzr.lemmatize(word)
    else: 
        lmtzr=lmtzr.lemmatize(word, pos)
    return lmtzr

def find_NP(input, NP_list, is_NP=False):
    '''
    find_longestNP 내부에서 쓰임'''
    if is_NP==False:
        if type(input)==list:

            if len(input)==1:
                input=input[0]
            
            tag=input[0][0]
            new_input=input[1:]

            if tag=='NP':
                is_NP=True
                find_NP(new_input, NP_list, is_NP)

            else:
                for i in new_input:
                    find_NP(i, NP_list)
    else:
        NP_list.append(input)

def extract_NP(input, sentence:list, is_sentence=False):
    '''
    find_longestNP 내부에서 쓰임'''
    is_tuple=True
    is_str=False
    if is_sentence==False:
        for i in input: ## i 중 tuple이 아닌게 하나라도 있으면
            if type(i)!=tuple:
                is_tuple=False
            if type(i)==str:
                is_str=True
    
        ## tuple로만 이루어진 경우[(tag),(word)]
        if is_tuple==True:
            word=input[1][0]
            is_sentence=True
            extract_NP(word, sentence, is_sentence)

        else:
            if is_str==False:   ## 아직 최소단위가 아닌 경우
                for i in input: ## 최소단위를 계속해서 찾아가야함
                    extract_NP(i, sentence)
    else: 
        sentence.append(input)

def find_longestNP(input:list)->str:
    '''
    9
    stanfordNLP 파싱결과를 리스트로 바꾼 인풋을 넣으면
    가징 긴 명사구 찾아줌->빈칸역할
    '''
    NP_list=[]
    find_NP(input, NP_list)
    NP_list.sort(key=len, reverse=True)
    longest_NP=NP_list[0]
    word_list=[]
    extract_NP(longest_NP, word_list)
    # print(word_list)
    sentence = " ".join(word_list)
    return sentence

def parsing_tolist(sexp:str)->list:
    '''
    stanfordnlp Dependency Parsing
    str->list
    '''
    stack, i, length = [[]], 0, len(sexp)
    while i < length:
        c = sexp[i]

        # print('c: ', c)
        # print('stack:', stack)
        reading = type(stack[-1])
        if reading == list:
            if   c == '(': stack.append([])
            elif c == ')': 
                stack[-2].append(stack.pop())
                if stack[-1][0] == ('quote',): stack[-2].append(stack.pop())
            elif c == '"': stack.append('')
            elif c == "'": stack.append([('quote',)])
            elif c in whitespace: pass
            else: stack.append((c,))
        elif reading == str:
            if   c == '"': 
                stack[-2].append(stack.pop())
                if stack[-1][0] == ('quote',): stack[-2].append(stack.pop())
            elif c == '\\': 
                i += 1
                stack[-1] += sexp[i]
            else: stack[-1] += c
        elif reading == tuple:
            if c in atom_end:
                atom = stack.pop()
                if atom[0][0].isdigit(): stack[-1].append(eval(atom[0]))
                else: stack[-1].append(atom)
                if stack[-1][0] == ('quote',): stack[-2].append(stack.pop())
                continue
            else: stack[-1] = ((stack[-1][0] + c),)
        i += 1
    return stack.pop()
