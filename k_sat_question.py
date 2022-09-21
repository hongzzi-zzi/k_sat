#%%
## import package
import json
import random
import warnings

from k_sat_func import *

warnings.filterwarnings(action='ignore')
#%%
f = open("/home/K_sat_final/testset/3.txt","r")
passageID=3
passage = f.read()
#%%
question_dict_sample={'passageID':None,
                    'question_type':None,
                    'question':None, 
                    'new_passage':None,
                    'answer':None,
                    'd1':None, 'd2':None, 'd3':None, 'd4':None}
#%% 18, 20, 22(목적/요지/주제): 한국어 보기, 23, 24, 41(주제/제목): 영어 보기
## 제목을 없애자 ㅋ
class Q1:
    def __init__(self):
        self.question_type=1
        # self.qlist=['목적으로', '주장으로', '요지로', '제목으로']
        self.qlist=['목적으로', '주장으로', '요지로']
        self.question=f'다음 글의 {random.choice(self.qlist)} 가장 적절한 것은?'

    def summarize(self, passage:str, num_sentence=5)->list:
        return summary(passage, num_sentences=num_sentence)

    def paraphrase(self, summary:list):
        paraphrase=paraphrasing_by_transe(summary)  ## list
        sent_completion_dict=get_sentence_completions(paraphrase)   ## dict
        return paraphrase, sent_completion_dict

    def get_false_paraphrase(self, sent_completion_dict:dict)->list:
        false_paraphrase=[]
        false_paraphrase_cnt = 1
        for key_sentence in sent_completion_dict:
            if false_paraphrase_cnt == 6:
                break
            partial_sentences = sent_completion_dict[key_sentence]
            false_sentences =[]
            false_sents = []
            for partial_sent in partial_sentences:
                for repeat in range(10):
                    false_sents = generate_sentences(partial_sent, key_sentence)
                    if false_sents != []:
                        break
                false_sentences.extend(false_sents)
            false_paraphrase.extend(paraphrasing_by_transe(false_sentences[:1]))
            false_paraphrase_cnt += 1
        return false_paraphrase

    def make_json(self, passageID:int, passage:str, is_Korean=False):
        question_dict=question_dict_sample.copy()
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=self.question_type
        question_dict['question'] = self.question
        
        summarize=self.summarize(passage)   ## list
        paraphrase, completion_dict=self.paraphrase(summarize)  ## list, dict
        false_paraphrase=self.get_false_paraphrase(completion_dict) ## list

        ## 랜덤으로 답 뽑기
        num=random.randint(0, 4)
        answer=paraphrase[num]  ## str
        dist_list=false_paraphrase[:num]+false_paraphrase[num+1:]   ## list

        ## 제목
        if '제목' in self.question:
            
            # if type(get_keyword_list(answer, max_word_cnt=5, top_n=1))==str:

            answer=sum(get_keyword_list(answer, max_word_cnt=5, top_n=1) , [])   ## input: str -> output: str
            dist_list=sum(get_keyword_list(dist_list, max_word_cnt=5, top_n=1), [])  ## input: list -> output: list
            is_Korean==False

        ## 번역
        if is_Korean==True:
            answer=transe_kor([answer])   ## input list로 맞춰주기
            dist_list=transe_kor(dist_list) ## input: list -> output: list
    
        question_dict['answer']=answer
        question_dict['d1']=dist_list[0]
        question_dict['d2']=dist_list[1]
        question_dict['d3']=dist_list[2]
        question_dict['d4']=dist_list[3]

        return json.dumps(question_dict, ensure_ascii = False)
#%% 26-28, 45(내용 일치/불일치): 영어 보기/한글 보기
class Q2:
    def __init__(self):
        self.question_type=2
        self.qlist=['적절한', '적절하지 않은']
        self.question=f'윗글에 관한 내용으로 가장 {random.choice(self.qlist)} 것은?'

    def summarize(self, passage:str, num_sentence=5)->list:
        return summary(passage, num_sentences=num_sentence)

    def paraphrase(self, summary:list)->list:
        paraphrase=paraphrasing_by_transe(summary)
        return paraphrase

    def get_false_sentence(self, paraphrase:list)->list:
        distractors=[]
        distractor_cnt = 1
        for key_sentence in paraphrase:
            if distractor_cnt == 6:
                break
            # print(type(key_sentence))   ## str
            keyword_list=get_keyword_list(key_sentence, max_word_cnt=2, top_n=1) 
            keyword=keyword_list[0]
            # print(type(keyword), keyword)# list

            # if len(keyword.split())==1:
            if len(keyword)==1:
                change_word=keyword[0]
            else:
                change_word=random.choice(keyword.split())[0]


            # print(type(change_word), change_word)# str
            antonym_list=get_antonym_list(get_lmtzr(change_word, get_pos(change_word)),1)
            
            if len(antonym_list)>0:
                distractor_sentence=key_sentence.replace(change_word, antonym_list[0])
            else: print('Q2: get_false_sentence ===> error');return None

            distractors.append(distractor_sentence)
            distractor_cnt += 1
        return distractors

    def make_json(self, passageID:int, passage:str, is_Korean=False):
        question_dict=question_dict_sample.copy()
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=self.question_type
        question_dict['question'] = self.question

        summarize=self.summarize(passage)   ## list
        key_sentences=self.paraphrase(summarize)    ## list
        false_sentences=self.get_false_sentence(key_sentences)  ## list

        ## 적절하지 않은? 적절한?
        if '않은' in self.question:   ## 적절하지 않은 것 고르기: 적절 4 안적절 1
            ans_list=false_sentences.copy()
            dist_list=key_sentences.copy()
        else:   ## 적절한 것 고르기: 적절 1 안적절 4
            ans_list=key_sentences.copy()
            dist_list=false_sentences.copy()

        ## 랜덤으로 답 뽑기
        num=random.randint(0, 4)# a <= num <= b
        answer=ans_list[num]    ## 정답
        del dist_list[num]  ## 오답 list 수정

        ## 번역
        if is_Korean==True:
            # print('Kor')
            answer=transe_kor([answer])   ## input list로 맞춰주기
            dist_list=transe_kor(dist_list) ## input: list -> output: list

        question_dict['answer']=answer[0]
        question_dict['d1']=dist_list[0]
        question_dict['d2']=dist_list[1]
        question_dict['d3']=dist_list[2]
        question_dict['d4']=dist_list[3]

        return json.dumps(question_dict, ensure_ascii = False)
#%% 36-37, 43(순서(ABC)): 영어 보기
class Q3:
    def __init__(self):
        self.question_type=3
        self.question='주어진 글 다음에 이어질 글의 순서로 가장 적절한 것을 고르시오.'

    def separate(self,passage:str):
        ## 문장단위로 쪼개고 ABC를 랜덤하게 234번째 문단 앞에 붙히기
        ## 4문단으로 분리, 1문단은 주어짐. 234 랜덤으로 ABC 부여
        ## 답안 생성, A B C 랜덤하게 배치, 서로 겹치지 않게
        # temp1=passage        #passage는 그대로 냅둘라고
        temp=passage.split('.') # 마침표 기준. 리스트로 쪼갬
        l=len(temp)
        # print(temp)
        num=l//4
        new_passage_lst=[]
        new_passage_lst.append(temp[:num])   ## 0문단-> 얘는 처음에 주어짐. 처음~1/4까지
        new_passage_lst.append(temp[num:2*num])     ## 1문단-> 1/4 다음 문장에서 1/2문장까지
        new_passage_lst.append(temp[2*num:3*num])   #~
        new_passage_lst.append(temp[3*num:])   #~


        # new_passage_lst[0]=sum(new_passage_lst[0][0],new_passage_lst[0][1])
        # print(new_passage_lst)
        for i in range(4):
            new_passage_lst[i]=' '.join(new_passage_lst[i])

        #정답을 랜덤하게 배치
        answer_list=[['A','B','C'],['A','C','B'],['B','A','C'],['B','C','A'],['C','A','B'],['C','B','A']]
        select=random.sample(answer_list,5)  #위에 6개 중 5개 랜덤하게 선택됨
        ans=select[0]   #첫번째 거가 정답임
        ans_num=[]
        for j in range(0,3):
            if ans[j]=='A': ans_num.append(1)
            if ans[j]=='B': ans_num.append(2)
            if ans[j]=='C': ans_num.append(3)

        distractors=select[1:5]   #나머지가 distractor 됨

        show=[]                 #정답 순서 따라 new_passage_lst를 show에 재배치
        for j in range(len(new_passage_lst)):
            show.append(new_passage_lst[j])

        show[ans_num[0]]=new_passage_lst[1]
        show[ans_num[1]]=new_passage_lst[2]
        show[ans_num[2]]=new_passage_lst[3]

        new_passage=''
        new_passage=str(show[0])+'\n'+'(A)'+'\t'+str(show[1])+'\n'+'(B)'+'\t'+str(show[2])+'\n'+'(C)'+'\t'+str(show[3])
             

        return new_passage, ans, distractors    ## str, str, list

    def make_json(self, passageID:int, passage:str)->dict:
        question_dict=question_dict_sample.copy()
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=self.question_type
        question_dict['question'] = self.question

        new_passage, ans, distractor=self.separate(passage)

        question_dict['new_passage'] = new_passage
        question_dict['answer']=ans
        question_dict['d1']=distractor[0]   ## 이게 출력될 때 A B C 이렇게 나올텐데 A->B->C로 애초에 저장할까?
        question_dict['d2']=distractor[1]
        question_dict['d3']=distractor[2]
        question_dict['d4']=distractor[3]

        return json.dumps(question_dict, ensure_ascii = False)
#%% 31(빈칸추론(단어)): 영어 보기
class Q4:
    def __init__(self):
        self.question_type=4
        self.question='다음 빈칸에 들어갈 말로 가장 적절한 것을 고르시오'

    def summarize(self, passage:str, num_sentence=1)->list: ## 중심문장 1문장(지문 내부 문장)
        return summary(passage, num_sentences=num_sentence)
    
    def paraphrase(self, summary:list)->list:   ## 1문장
        return paraphrasing_by_transe(summary)

    def get_answer(self, paraphrase:list)->str: ## 1문장 들어가서 키워드 1개 나옴
        answer_list=sum(get_keyword_list(paraphrase, max_word_cnt=1, top_n=1), [])   #[['keyword']] 
        return answer_list[0]    ## str

    def get_distractors(self, answer:str)->list:    ## 오답 단어 4개 만들기
        ans_lmtzr=get_lmtzr(answer, get_pos(answer))
        antonym_list=get_antonym_list(ans_lmtzr, 7)
        synonym_list=get_synonym_list(ans_lmtzr, 7)

        distractors=del_same_lemmatization(antonym_list+synonym_list)

        ## 정확도순으로 해보려했는데 별 의미가 없다....
        '''distractor_tuple=[]
        for i in distractors:
            similarity=word_similarity(answer, i)
            distractor_tuple.append((i, similarity))
        sorted(distractor_tuple, key=lambda distractor_tuple: distractor_tuple[1])
        print(distractor_tuple)'''

        distractors = random.sample(distractors, 4)
        return distractors

    def make_new_passage(self, passage:str, summarize:str, paraphrase:str, answer:str)->str: #fin, str/None 리턴
        if summarize in passage:
            space='_'*int(len(answer)*0.6)
            new_passage=passage.replace(summarize, paraphrase.replace(answer, space))
            return new_passage  ## str
        else: print('No result');return None

    def make_json(self, passageID:int, passage:str):
        question_dict=question_dict_sample.copy()
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=self.question_type
        question_dict['question'] = self.question   ## '다음 빈칸에 들어갈 말로 가장 적절한 것을 고르시오'

        summarize=self.summarize(passage)  ## list
        paraphrase=self.paraphrase(summarize)   ## list
        answer=self.get_answer(paraphrase)  ## str
        dist_list=self.get_distractors(answer)    ## list
        new_passage=self.make_new_passage(passage, summarize[0], paraphrase[0], answer)   ##str

        ## 새로운 passage
        question_dict['new_passage']=new_passage

        question_dict['answer']=answer
        question_dict['d1']=dist_list[0]
        question_dict['d2']=dist_list[1]
        question_dict['d3']=dist_list[2]
        question_dict['d4']=dist_list[3]

        
        return json.dumps(question_dict, ensure_ascii = False)
#%% 30, 42(적절하지 않은 단어)
class Q5:
    def __init__(self):
        self.question_type=5
        self.question='다음 글의 밑줄 친 부분 중, 문맥상 낱말의 쓰임이 적절하지 않은 것은?'

    ## passage 들어가서 단어 5개 나오는 걸로 ...
    def get_keyword(self, passage:str)->list: ## 5문장 들어가서 단어 5개 나옴
        ## [['w1'],['w2'], ...] -> ['w1','w2', ...] 변환
        keyword_list=get_keyword_list(passage, max_word_cnt=1, top_n=5)
        return keyword_list[0]

    def get_keyword_antonym(self, keyword_list:list)->list: ## 단어 5개 들어가서 단어 5개 나옴
        antonym_list=[]
        # print(keyword_list)
        for kwd in keyword_list:
            # print(kwd, type(kwd))
            antonym_list=antonym_list+get_antonym_list(get_lmtzr(kwd, get_pos(kwd)), 1)
        return antonym_list

    def make_new_passage(self, passage:str, origin_list:list, new_list: list):# new passage(Str), ans(int)
        new_passage=''+passage

        ## 적절하지 않은 단어를 고르는게 답인데 적절하지 않은 단어를 못찾았으면 답을 구할 때 빼라
        ex=[1, 2, 3, 4, 5]
        none_list=[]
        if 'None' in new_list:
            none_list.append(new_list.index('None'))
        for i in none_list:
            ex.remove(i)
        tmp_answer = random.choice(ex)

        ## new_list 변경
        for i in range(len(origin_list)):
            if i==tmp_answer-1: ## 단어가 바뀌는 경우
                ans_word=new_list[i]
                new_list[i]='(num) '+new_list[i]
                # print(tmp_answer, ans_word)
            else: 
                new_list[i]='(num) '+origin_list[i]

        # print(new_list)

        for i in range(len(origin_list)):
            cnt_passage=passage.count(origin_list[i])
            if cnt_passage==1:
                new_passage=new_passage.replace(origin_list[i], new_list[i], 1)
            else:
                ans=random.randint(1, cnt_passage)
                new_passage=new_passage.replace(origin_list[i], new_list[i], ans)
                new_passage=new_passage.replace(new_list[i], origin_list[i], ans-1)

        num=1
        answer=0
        for i in range(len(origin_list)):
            pause='(%d) '%(num)
            new_passage=new_passage.replace('(num) ', pause, 1)
            # print(pause)
            word=new_passage.split(pause)[-1].split()[0]
            # print(word)
            if word==ans_word:
                answer=num
            num=num+1

        return new_passage, answer
    
    def make_json(self, passageID:int, passage:str):
        question_dict=question_dict_sample.copy()
        flag=True
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=self.question_type
        question_dict['question'] =self.question    ## '다음 글의 밑줄 친 부분 중, 문맥상 낱말의 쓰임이 적절하지 않은 것은?'
        keyword=self.get_keyword(passage) ## list: ['w1','w2', ...]
        keyword_antonym=self.get_keyword_antonym(keyword)   ## list: ['w1','w2', ...]

        ans_list=keyword_antonym    ## 틀린 단어(바뀌어야 함)
        dist_list=keyword    ## 원래 단어(바뀌지 말아야 함)

        # 새로운 passage 만들기, answer
        new_passage, answer= self.make_new_passage(passage, dist_list, ans_list)

        question_dict['new_passage']=new_passage
        question_dict['answer']=answer

        ex=[1, 2, 3, 4, 5]
        ex.remove(answer)
        
        question_dict['d1']=ex[0]
        question_dict['d2']=ex[1]
        question_dict['d3']=ex[2]
        question_dict['d4']=ex[3]

        return json.dumps(question_dict, ensure_ascii = False)
#%% 38-39 문장이 들어가기에 적절한 곳
class Q6:
    def __init__(self):
        self.question_type=6
        self.question='글의 흐름으로 보아, 주어진 문장이 들어가기에 가장 적절한 곳을 고르시오.'
    
    #문장단위로 쪼개기
    def separate(self,passage:str):
        temp=passage.split('.') # 마침표 기준. 리스트로 쪼갬
        del temp[len(temp)-1]
        l=len(temp)

        # 정답 고르고 distractor 생성
        answer_list=[1,2,3,4,5]
        ans=random.randint(1,5)

        #문장 번호
        num=range(1,l-1)          # 문장 번호
        select=random.sample(num,5) # 그 중에 5개
        select.sort()    

        distractors=[x for x in answer_list if x!=ans] 

        # 정답 문장
        ans_text=temp[select[ans-1]]
        # print('정답번호: '+str(ans))              ##정답의 번호
        # print('문장번호: '+str(select[ans-1]))    ##정답문장의 문장번호
        # print(ans_text)

        ans_text_num=select[ans-1]


        head=range(1,ans_text_num)
        tail=range(ans_text_num+2,l)

        ## 여기서 부터 case 분류 (1번, 234번, 5번)
        if (ans==1):
            tail_select=random.sample(tail,4)
            tail_select.sort()
            select=tail_select
            select.append(ans_text_num)
        if(ans==5):
            head_select=random.sample(head,4)
            head_select.sort()
            select=head_select
            select.append(ans_text_num)
        if(ans==2 or ans==3 or ans==4):
            head_select=random.sample(head,ans-1)
            head_select.sort()
            tail_select=random.sample(tail,5-ans)
            tail_select.sort()
            select=head_select+tail_select
            select.append(ans_text_num)
        
        select.sort()

        for i in range(0,5):
            temp[select[i]]='('+str(answer_list[i])+')'+str(temp[select[i]])
        
        temp[select[ans-1]]='('+str(ans)+')'+str(temp[select[ans-1]+1])
        del temp[select[ans-1]+1]


        ## 문제 출력
        new_passage='. '.join(temp)
        new_passage=str(ans_text)+'\n\n'+str(new_passage)+'.'

        return new_passage, ans, distractors


    def make_json(self, passageID:int, passage:str):# , passage:str)->dict:
        question_dict=question_dict_sample.copy()
        
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=self.question_type
        question_dict['question'] = self.question
        new_passage, ans, distractor=self.separate(passage)

        question_dict['new_passage'] = new_passage
        question_dict['answer']=ans
        question_dict['d1']=distractor[0]   
        question_dict['d2']=distractor[1]
        question_dict['d3']=distractor[2]
        question_dict['d4']=distractor[3]

        return json.dumps(question_dict, ensure_ascii = False)
#%% 35 전체 흐름과 관계 없는 문장
class Q7:
    def __init__(self):
        self.question_type=7
        self.question='다음 글에서 전체 흐름과 관계 없는 문장은?'

    def summarize(self, passage:str, num_sentence=5)->list:
        return summary(passage, num_sentences=num_sentence)

    def get_completion_dict(self, summary:list)->dict:
        paraphrase=paraphrasing_by_transe(summary)  ## list
        sent_completion_dict=get_sentence_completions(paraphrase)## dict
        return sent_completion_dict

    def get_false_paraphrase(self, sent_completion_dict:dict)->list:
        false_paraphrase=[]
        false_paraphrase_cnt = 1
        for key_sentence in sent_completion_dict:
            if false_paraphrase_cnt == 6:
                break
            partial_sentences = sent_completion_dict[key_sentence]
            false_sentences =[]
            false_sents = []
            for partial_sent in partial_sentences:
                for repeat in range(10):
                    false_sents = generate_sentences(partial_sent, key_sentence)
                    if false_sents != []:
                        break
                false_sentences.extend(false_sents)
            false_paraphrase.extend(paraphrasing_by_transe(false_sentences[:1]))
            false_paraphrase_cnt += 1
        return false_paraphrase

    def make_new_passage(self, passage:str, origin_list:list, false_list:list, answer:int)->str:
        new_passage=''+passage
        for i in range(len(origin_list)):
            origin=origin_list[i]

            if i+1==answer:
                new='('+str(i+1)+') '+false_list[i]
            else:
                new='('+str(i+1)+') '+origin
            
            if origin in passage:
                new_passage=new_passage.replace(origin, new)
            else:
                new_passage=new_passage.lower().replace(origin.lower(), new)

        return new_passage

    def make_json(self, passageID:int, passage:str)->dict:
        question_dict=question_dict_sample.copy()
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=self.question_type
        question_dict['question'] = self.question   ## 다음 글에서 전체 흐름과 관계 없는 문장은?

        summarize=self.summarize(passage)  ## list
        sent_completion_dict=self.get_completion_dict(summarize)    ## dict
        false_paraphrase=self.get_false_paraphrase(sent_completion_dict)    ## list

        ## 랜덤으로 답 뽑기
        ex=[1, 2, 3, 4, 5]
        answer = random.choice(ex)
        ex.remove(answer)

        new_passage=self.make_new_passage(passage, summarize, false_paraphrase, answer)
        question_dict['new_passage']=new_passage
    
        question_dict['answer']=str(answer)
        question_dict['d1']=str(ex[0])
        question_dict['d2']=str(ex[1])
        question_dict['d3']=str(ex[2])
        question_dict['d4']=str(ex[3])

        return json.dumps(question_dict, ensure_ascii = False)
#%% 40 글의 내용 요약하고 빈칸 2개 단어 고르기
## https://huggingface.co/docs/transformers/v4.21.0/en/main_classes/pipelines#transformers.SummarizationPipeline
class Q8:
    def __init__(self):
        self.question_type=8
        self.question='다음 글의 내용을 요약하고자 한다. 빈칸 (A), (B)에 들어갈 말로 가장 적절한 것은?'

    def summarize(self, passage:str, num_sentence=2)->list:
        return summary(passage, num_sentences=num_sentence)

    def paraphrase(self, summary:list)->list:
        return paraphrasing_by_transe(summary)

    def get_keyword(self, paraphrase:list, option='Q8') ->list:
        if option=='Q8':    ## option=='Q8' 인 경우 무조건 top_n==2
            top_n=2
        keyword_list= get_keyword_list(paraphrase, max_word_cnt=1, top_n=top_n, option=option)
        return keyword_list

    def get_distractors_fromWord(self, keyword:str, paraphrase:str)->list:    ## 오답 단어 2개 만들기
        word_lmtzr=get_lmtzr(keyword, get_pos(keyword))
        synonym_list=del_same_lemmatization(get_synonym_list(word_lmtzr, 3))
        antonym_list=del_same_lemmatization(get_antonym_list(word_lmtzr, 3))

        for i in synonym_list:
            if i in paraphrase:
                synonym_list.remove(i)
        for i in antonym_list:
            if i in paraphrase:
                antonym_list.remove(i)

        if len(synonym_list)>0 and len(antonym_list)>0:return [synonym_list[0],antonym_list[0]]
        elif len(synonym_list)==0 and len(antonym_list)>=2: return[antonym_list[0], antonym_list[1]]
        elif len(synonym_list)>=2 and len(antonym_list)==0:return [synonym_list[0], synonym_list[1]]
        else:return None
    
    def get_distractors_fromPassage(self, passage:str, keyword:str, paraphrase:str)->list:    ## 오답 단어 2개 만들기
        # print('616')
        kwd_list=get_keyword_list(passage, 1, 15)## [['robots', 'factories', 'workers', 'employees', 'labor', 'planning', 'manufacturing', 'employment', 'management', 'retraining', 'motions', 'assembly', 'fear', 'humans', 'human']]
        # print(kwd_list[0], type(kwd_list[0]))## 그냥리스트
        # print(kwd_list, type(kwd_list))## 이중리스트
        passage_keyword=del_same_lemmatization(kwd_list[0])
        # print('asdfasdfadsfasdfasdfasd')
        # print(paraphrase)
        print(type(passage_keyword))
        new_passage_keyword=[]

        for i in passage_keyword:
            if i in paraphrase or get_lmtzr(i, get_pos(i)) in paraphrase:print('del '+i)
            else:new_passage_keyword.append(i)

        # print(new_passage_keyword)
        
        print(keyword, type(keyword))

        result=[]
        kwd_lmtzr=get_lmtzr(keyword, get_pos(keyword))
        
        for word in new_passage_keyword:
            word_lmtzr=get_lmtzr(word, get_pos(keyword))
            if word!=keyword and word_lmtzr!=kwd_lmtzr: 
                result.append(word)
        if len(result)>=4: return result[:4]
        else:return None

    def get_distractors(self, passage:str, keyword:list, paraphrase:list)->list:
        a, b=keyword[0], keyword[1]## str
        ## list->str
        paraphrase_str=''
        for sentence in paraphrase:
            paraphrase_str=paraphrase_str+sentence
            

        a_distractors=self.get_distractors_fromWord(a, paraphrase_str)  ## 2개
        # print('a_dis: ', a_distractors)
        # print(type(b), type(paraphrase_str))# str, str
        b_distractors=self.get_distractors_fromPassage(passage, b, paraphrase_str) ## 4개

        if a_distractors==None:
            a_distractors=self.get_distractors_fromPassage(passage, a, paraphrase_str)  # 4개
            b_distractors=self.get_distractors_fromWord(b, paraphrase_str)  ## 2개

        if a_distractors==None or b_distractors==None:
            print("Q8: get_distractors: fail to get distractors...")
            return None
        
        return a_distractors, b_distractors
        
    def make_new_passage(self, passage:str, paraphrase:list, keyword:list)->str:
        new_passage=passage+'\n\n==>'
        new_paraphrase=[]

        for i in range(len(paraphrase)):
            if keyword[i] in paraphrase[i]:
                if i==0:space='__(A)__'
                elif i==1:space='__(B)__'
                else:space='_____'  ## 이게 될 일은 없을걸...?
                new_paraphrase.append(paraphrase[i].replace(keyword[i],space))

        for sentence in new_paraphrase:
            new_passage=new_passage+' '+str(sentence)
        return new_passage

    def make_json(self, passageID:int, passage:str):
        question_dict=question_dict_sample.copy()
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=self.question_type
        question_dict['question'] = self.question
        summarize=self.summarize(passage)   ## list
        paraphrase=self.paraphrase(summarize)   ## list
        keyword=self.get_keyword(paraphrase)    ## list
        new_passage=self.make_new_passage(passage, paraphrase, keyword) ## str
        a_distractors, b_distractors =self.get_distractors(passage, keyword, paraphrase)
        
        distractors=[]
        if len(a_distractors)==2:
            distractors.append('(A)'+keyword[0]+' (B)'+b_distractors[0])
            distractors.append('(A)'+a_distractors[0]+' (B)'+b_distractors[1])
            distractors.append('(A)'+a_distractors[0]+' (B)'+b_distractors[2])
            distractors.append('(A)'+a_distractors[1]+' (B)'+b_distractors[3])
        elif len(b_distractors)==2:
            distractors.append('(A)'+a_distractors[0]+' (B)'+keyword[1])
            distractors.append('(A)'+a_distractors[1]+' (B)'+b_distractors[0])
            distractors.append('(A)'+a_distractors[2]+' (B)'+b_distractors[0])
            distractors.append('(A)'+a_distractors[3]+' (B)'+b_distractors[1])

        question_dict['new_passage']=new_passage
        question_dict['answer']='(A)'+keyword[0]+' (B)'+keyword[1]

        question_dict['d1']=distractors[0]
        question_dict['d2']=distractors[1]
        question_dict['d3']=distractors[2]
        question_dict['d4']=distractors[3]
        return json.dumps(question_dict, ensure_ascii = False)
#%% 32-34 빈칸추론 (구, 절)-> 실험중..
PATH='/home/stanford-corenlp-4.5.0'

os.chdir(PATH)
os.system('pwd')
os.system('java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
                    -preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
                    -status_port 9000 -port 9001 -timeout 15000 &')
os.chdir('/home/K-sat')
nlp = StanfordCoreNLP('http://localhost', port=9001, memory='8g')
cc = OpenCC('t2s')

atom_end = set('()"\'') | set(whitespace)
class Q9:
    def __init__(self):
        self.question_type=9
        self.question='다음 빈칸에 들어갈 말로 가장 적절한 것을 고르시오.'

    ## 중심문장
    def summarize(self, passage:str, num_sentence=1)->list:
        return summary(passage, num_sentences=num_sentence)

    ## 답 찾기: 한문장 str로 들어가서 str로 리턴
    def get_answer(self, summarize:str)->str:
        parsing_res=parsing_tolist(nlp.parse(summarize))
        ans=find_longestNP(parsing_res)
        return ans

    ## 오답 4개 생성
    def get_distractors(self):
        None
    
    def make_newpassage(self, passage:str, sentence:str)->str:
        space='_'*int(len(sentence)*0.6)
        return passage.replace(sentence, space)

    def make_json(self, passageID:int, passage:str):
        question_dict=question_dict_sample.copy()
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=self.question_type
        question_dict['question'] = self.question
        
        summarize=self.summarize(passage)   ## list(1문장 나오니까 답 구할때 summarize[0])
        answer=self.get_answer(summarize[0])
        question_dict['new_passage']=self.make_newpassage(passage, answer)
        question_dict['answer']=str(answer)
        # question_dict['d1']=str(ex[0])
        # question_dict['d2']=str(ex[1])
        # question_dict['d3']=str(ex[2])
        # question_dict['d4']=str(ex[3])


        return json.dumps(question_dict, ensure_ascii = False)
q9=Q9()
q9_json=q9.make_json(passageID, passage)
print(q9_json)

#%%
# for test
q1=Q1()
q1_json_eng=q1.make_json(passageID, passage, is_Korean=False)
q1_json_kor=q1.make_json(passageID, passage, is_Korean=True)
print(q1_json_eng)
print(q1_json_kor)
#%%
# 26-28, 45(내용 일치/불일치): 영어 보기/한국어보기
q2=Q2()
q2_json_eng=q2.make_json(passageID, passage, is_Korean=False)
print(q2_json_eng)
q2_json_kor=q2.make_json(passageID, passage, is_Korean=True)
print(q2_json_kor)
#%%
# 36-37, 43(순서(ABC)): 영어 보기
q3=Q3()
q3_json=q3.make_json(passageID, passage)
print(q3_json)
#%%
# 31(빈칸추론(단어)): 영어 보기
q4=Q4()
q4_json=q4.make_json(passageID, passage)
print(q4_json)
#%%
# 30, 42(적절하지 않은 단어)
q5=Q5()
q5_json=q5.make_json(passageID, passage)
print(q5_json)
#%%
# 38-39 문장이 들어가기에 적절한 곳 
q6=Q6()
q6_dict=q6.make_json(passageID, passage)
print(q6_dict)
#%%
# 35 전체 흐름과 관계 없는 문장
q7=Q7()
q7_json=q7.make_json(passageID, passage)
print(q7_json)
#%%
# 40 글의 내용 요약하고 빈칸 2개 단어 고르기
q8=Q8()
q8_json=q8.make_json(passageID, passage)
print(q8_json)

# %%
q9=Q9()
q9_json=q9.make_json(passageID, passage)
print(q9_json)

# %%
